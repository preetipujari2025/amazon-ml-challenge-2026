"""
Business Address Normalization & Postal Code Extraction Module
Amazon ML Challenge 2026 - Business Entity Resolution (Member 1)

This module provides functions to clean and normalize raw business address strings
across multiple languages and scripts (English, French, Hindi, Kannada, Telugu),
as well as a high-precision extractor for postal/PIN codes without external dependencies.
"""

import math
import re
import unicodedata
from typing import Optional


def normalize_address(text) -> str:
    """
    Normalize a business address string.

    Pipeline:
      1. Type & null-safety check: Returns empty string for None, NaN, or non-string inputs.
      2. Unicode normalization (NFKC):
         - Resolves compatibility characters, fullwidth/halfwidth forms, and ligatures.
         - Preserves native characters across scripts (Latin with accents, Devanagari,
           Kannada, Telugu).
      3. Removal of invisible formatting characters:
         - Strips zero-width characters (ZWNJ \u200c, ZWJ \u200d, ZWSP \u200b, BOM \ufeff).
      4. Lowercasing:
         - Converts to lowercase using Unicode casefolding rules.
      5. Standardization of quotes and apostrophes:
         - Converts curly quotes to straight single quotes.
      6. Removal of noise brackets and symbols:
         - Cleans parentheses, brackets, angle brackets (<>), hashes (#), asterisks (*), etc.
      7. Punctuation normalization:
         - Collapses multiple dots, dashes, and commas.
         - Normalizes abbreviation dots (e.g. 'no.', 'st.', 'rd.') while preserving digits.
         - Cleans isolated dashes and slashes while preserving intra-word / compound hyphens
           (e.g. 'hauts-de-france', 'e-7') and municipal slashes ('164/b-43', '83/1').
         - Normalizes comma spacing to a clean single comma and space (', ').
      8. Whitespace cleanup:
         - Strips leading/trailing punctuation and collapses multiple spaces.

    Parameters:
      text: Input address string (str, or None/NaN)

    Returns:
      Cleaned and normalized address string (str)
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        if isinstance(text, float) and math.isnan(text):
            return ""
        try:
            s_val = str(text)
            if s_val.strip().lower() in ("nan", "none", "null"):
                return ""
            text = s_val
        except Exception:
            return ""

    if not text.strip():
        return ""

    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip invisible formatting characters
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)

    # 3. Unicode-aware lowercasing
    text = text.lower()

    # 4. Standardize quotes and apostrophes
    text = re.sub(r"[\u2018\u2019\u02bc`´]", "'", text)
    text = re.sub(r'[\u201c\u201d"]', " ", text)

    # 5. Remove noise brackets and symbols (<>, (), [], {}, #, *, ~, etc.)
    text = re.sub(r"[<>()\[\]{}#~*_!?;]", " ", text)

    # 6. Normalize multiple dots, dashes, commas
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"-{2,}", " ", text)
    text = re.sub(r",{2,}", ",", text)

    # 7. Normalize periods in abbreviations (e.g. 'no.' -> 'no', 'st.' -> 'st', trailing dots)
    text = re.sub(r"\.(?=\s|,|$)", "", text)

    # 8. Clean isolated dashes and slashes:
    # - Remove isolated dashes with spaces around them (' - ')
    # - Remove leading dash before numbers (e.g. ' -570' -> ' 570')
    # - Keep intra-word hyphens ('hauts-de-france', 'e-7') and slashes ('164/b-43', '83/1')
    text = re.sub(r"\s+-\s+", " ", text)
    text = re.sub(r"(?<=\s)-(?=[0-9])", "", text)
    text = re.sub(r"(?<=\s)[-/](?=\s)", " ", text)

    # 9. Clean isolated apostrophes
    text = re.sub(r"(?<![^\W_])'|'(?![^\W_])", " ", text)

    # 10. Standardize comma spacing (ensure single comma followed by space)
    text = re.sub(r"\s*,\s*", ", ", text)

    # 11. Strip leading/trailing commas and whitespace
    text = re.sub(r"^[,\s]+|[,\s]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_pin_code(text) -> Optional[str]:
    """
    Extract a postal or PIN code from an address string if clearly present.

    Rules:
      - Looks for explicit keywords (pin, pincode, zip, zipcode, postal code, code postal)
        with 5-6 digits.
      - Looks for standalone 6-digit Indian PIN codes ([1-9][0-9]{5}) that are not functioning
        as house, street, plot, or rural road numbers.
      - Looks for standalone 5-digit French postal codes or US ZIP codes that are not functioning
        as house, street, suite, or box numbers.
      - Returns None if not confidently detectable (does not guess or infer missing codes).

    Parameters:
      text: Input address string (str, or None/NaN)

    Returns:
      5 or 6 digit postal code string (str), or None
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    # 1. Explicit keyword match (PIN, pincode, zip, zipcode, postal code, code postal)
    kw_match = re.search(
        r"\b(?:pin|pincode|pin\s*code|zip|zipcode|zip\s*code|postal\s*code|code\s*postal)\s*[:\-#]?\s*([0-9]{5,6})\b",
        text,
        re.IGNORECASE,
    )
    if kw_match:
        return kw_match.group(1)

    # 2. Standalone 6-digit Indian PIN code (digits 1-9 followed by 5 digits)
    matches_6 = list(re.finditer(r"\b([1-9][0-9]{5})\b", text))
    for m in reversed(matches_6):
        start_pos = m.start()
        prefix = text[:start_pos].strip()
        suffix = text[m.end():].strip().lower()
        # If at very beginning of address followed by road/st/hwy, it's a road/house number
        if start_pos == 0 and re.match(
            r"^(?:\d+\s+)?(?:rd|road|hwy|highway|street|st|ave|blvd|lane|ct)\b",
            suffix,
        ):
            continue
        # If preceded by 'flat', 'plot', 'survey', 'sy', 'house', 'h.no', 'po box'
        if re.search(
            r"\b(?:flat|plot|survey|sy|house|h\.no|po\s*box|road|rd|highway|hwy|route|rt)\s*(?:no\.?|#)?\s*[:\-]?\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        return m.group(1)

    # 3. Standalone 5-digit US ZIP or French Postal Code (e.g. 59000, 75001, 74464)
    matches_5 = list(re.finditer(r"\b([0-9]{5})(?:-[0-9]{4})?\b", text))
    for m in reversed(matches_5):
        start_pos = m.start()
        prefix = text[:start_pos].strip()
        suffix = text[m.end():].strip().lower()
        # If at very beginning of address, it is a house/street number (e.g. '17560 Ellis Road')
        if start_pos == 0:
            continue
        # If preceded by unit/suite/floor/box/plot/flat
        if re.search(
            r"\b(?:po\s*box|box|suite|ste|unit|apt|apartment|fl|floor|room|rm|plot|flat)\s*(?:no\.?|#)?\s*[:\-]?\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        # If followed immediately by street suffixes (e.g. '11852 16th St' or 'Ridge Road')
        if re.match(
            r"^[a-z0-9\-]+\s+(?:rd|road|st|street|ave|avenue|blvd|boulevard|lane|ln|drive|dr|way|court|ct|trail|highway|hwy|circle|cir|terrace|ter)\b",
            suffix,
        ):
            continue
        return m.group(1)

    return None


if __name__ == "__main__":
    import sys

    # Ensure UTF-8 output encoding in Windows console
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    test_cases = [
        # 1. English with extra punctuation, commas, multiple dots, trailing noise
        (
            "  << 1795 Westchester Drive,, High Point , NC...  ",
            "1795 westchester drive, high point, nc",
            None,
        ),
        # 2. English US address with ZIP code at end
        (
            "17560 Ellis Road, Tahlequah, OK 74464",
            "17560 ellis road, tahlequah, ok 74464",
            "74464",
        ),
        # 3. English US address with 5-digit house number but NO ZIP code
        (
            "17560 Ellis Road, Tahlequah, OK",
            "17560 ellis road, tahlequah, ok",
            None,
        ),
        # 4. English with ZIP+4 code
        (
            "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio 45206-1234",
            "1 ivanhoe ave, po box 6009, cincinnati, ohio 45206-1234",
            "45206",
        ),
        # 5. French address with postal code, accents, and hyphenated region
        (
            "63 R. DE DIEPPE, 59000 LILLE, Hauts-de-France",
            "63 r de dieppe, 59000 lille, hauts-de-france",
            "59000",
        ),
        # 6. French address with accents, avenue, and commune
        (
            "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
            "175 boulevard du président franklin roosevelt, bordeaux, nouvelle-aquitaine",
            None,
        ),
        # 7. French address with postal code and bis
        (
            "5 bis Rue Pierre Dignac, 33260 La Teste-de-Buch",
            "5 bis rue pierre dignac, 33260 la teste-de-buch",
            "33260",
        ),
        # 8. Hindi address with Devanagari script, hyphens, and 6-digit PIN code
        (
            "केएच नं. -570/13, नई दिल्ली, पश्चिम दिल्ली, दिल्ली - 110041",
            "केएच नं 570/13, नई दिल्ली, पश्चिम दिल्ली, दिल्ली 110041",
            "110041",
        ),
        # 9. Kannada address with native script, door no, and PIN code
        (
            "ಡೋರ್ ನಂ. 183, 41ನೇ ಕ್ರಾಸ್, ಬೆಂಗಳೂರು 560082, ಕರ್ನಾಟಕ",
            "ಡೋರ್ ನಂ 183, 41ನೇ ಕ್ರಾಸ್, ಬೆಂಗಳೂರು 560082, ಕರ್ನಾಟಕ",
            "560082",
        ),
        # 10. Telugu address with native script and slashes
        (
            "సర్వే నం. 83/1, రాయదుర్గం, హైదరాబాద్, తెలంగాణ",
            "సర్వే నం 83/1, రాయదుర్గం, హైదరాబాద్, తెలంగాణ",
            None,
        ),
        # 11. Address with flat number containing 5 digits (should NOT be detected as PIN)
        (
            "Prestige Falcon City, Kanakapura Main Road, Flat No. 51262, Bangalore",
            "prestige falcon city, kanakapura main road, flat no 51262, bangalore",
            None,
        ),
        # 12. Edge cases: None, NaN, and empty
        (None, "", None),
        (float("nan"), "", None),
        ("   ", "", None),
    ]

    print("=" * 75)
    print("TESTING normalize_address() AND extract_pin_code() ACROSS SAMPLES")
    print("=" * 75)

    all_passed = True
    for idx, (addr, exp_norm, exp_pin) in enumerate(test_cases, start=1):
        norm_res = normalize_address(addr)
        pin_res = extract_pin_code(addr)
        norm_ok = norm_res == exp_norm
        pin_ok = pin_res == exp_pin
        test_passed = norm_ok and pin_ok
        if not test_passed:
            all_passed = False

        status = "PASS" if test_passed else "FAIL"
        print(f"[{status}] Test #{idx:02d}:")
        print(f"       Raw Input:      {repr(addr)}")
        print(f"       Norm Output:    {repr(norm_res)}")
        print(f"       Norm Expected:  {repr(exp_norm)} (Match: {norm_ok})")
        print(f"       PIN Output:     {repr(pin_res)}")
        print(f"       PIN Expected:   {repr(exp_pin)} (Match: {pin_ok})")
        print("-" * 75)

    print(f"\nFinal Result: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}\n")
