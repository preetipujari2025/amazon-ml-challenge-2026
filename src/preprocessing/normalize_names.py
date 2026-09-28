"""
Business Name Normalization Module
Amazon ML Challenge 2026 - Business Entity Resolution (Member 1)

This module provides functions to clean and normalize raw business name strings
across multiple languages and scripts (English, French, Hindi, Kannada, Telugu).
"""

import math
import re
import unicodedata


def normalize_name(text) -> str:
    """
    Normalize a business name string.

    Pipeline:
      1. Type & null-safety check: Returns empty string for None, NaN, or non-string inputs.
      2. Unicode normalization (NFKC):
         - Resolves compatibility characters, fullwidth/halfwidth forms, and ligatures.
         - Preserves native characters across scripts (Latin with accents, Devanagari,
           Kannada, Telugu).
      3. Removal of invisible formatting characters:
         - Strips zero-width characters (ZWNJ \u200c, ZWJ \u200d, ZWSP \u200b, BOM \ufeff)
           which often cause identical-looking Indian script words to fail string equality.
      4. Lowercasing:
         - Converts to lowercase using Unicode casefolding rules (e.g. 'École' -> 'école').
      5. Punctuation normalization:
         - Normalizes curly quotes/apostrophes to standard single quotes.
         - Replaces noise punctuation (brackets, angle brackets, slashes, dashes, stray commas,
           multiple periods) with spaces while preserving intra-word apostrophes (e.g. "orelee's",
           "l'étoile") and business symbols like '&' and '+'.
         - Eliminates leading/trailing non-alphanumeric noise characters.
      6. Whitespace cleanup:
         - Collapses consecutive whitespace (spaces, tabs, newlines) into a single space and strips.

    Parameters:
      text: Input business name (str, or None/NaN)

    Returns:
      Cleaned and normalized business name string (str)
    """
    # 1. Null / None / NaN safety check
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

    # 2. Unicode normalization (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 3. Strip zero-width / invisible formatting characters common in multi-script input
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)

    # 4. Unicode-aware lowercasing
    text = text.lower()

    # 5. Punctuation normalization
    # Standardize quotation marks and apostrophes
    text = re.sub(r"[\u2018\u2019\u02bc`´]", "'", text)
    text = re.sub(r'[\u201c\u201d"]', " ", text)

    # Replace punctuation characters (commas, periods, semicolons, brackets, dashes, etc.)
    # with spaces while retaining:
    # - Letters (Unicode category L*)
    # - Combining Marks/Vowels (Unicode category M*, essential for Hindi/Kannada/Telugu matras)
    # - Numbers (Unicode category N*)
    # - Intra-word apostrophe (')
    # - Connecting business symbols ('&', '+')
    cleaned_chars = []
    for char in text:
        cat = unicodedata.category(char)
        if cat[0] in ("L", "M", "N"):
            cleaned_chars.append(char)
        elif char in ("'", "&", "+"):
            cleaned_chars.append(char)
        else:
            # Replace all other punctuation / symbols / delimiters with space
            cleaned_chars.append(" ")

    cleaned = "".join(cleaned_chars)

    # Remove isolated apostrophes, pluses, or ampersands that are not connecting characters
    # Note: [^\W_] matches any Unicode letter or digit (including French accents like é, à, ç)
    cleaned = re.sub(r"(?<![^\W_])'|'(?![^\W_])", " ", cleaned)
    cleaned = re.sub(r"(?<=\s)&(?=\s)|(?<=\s)\+(?=\s)", " ", cleaned)

    # 6. Collapse multiple spaces and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


if __name__ == "__main__":
    import sys

    # Ensure UTF-8 output encoding in Windows console
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    test_cases = [
        # 1. English with stray punctuation, multiple spaces, and legal suffix
        ("  << Orelee's Barbershop, Inc...  ", "orelee's barbershop inc"),
        # 2. English with abbreviation periods and mixed case
        ("Pvt. EFS Print Ventures Ltd.", "pvt efs print ventures ltd"),
        # 3. English with business symbol (&, +)
        ("B+ Retail Inc & Co.", "b+ retail inc co"),
        # 4. English with leading noise punctuation and trailing noise
        ("-- Holloway Peak Inc Seafood --", "holloway peak inc seafood"),
        # 5. French with accented characters, leading brackets, and punctuation
        ("  << Team  École & Co., Ltd... ", "team école co ltd"),
        # 6. French with accents, acronym periods, and hyphenated names
        (
            "Fractales Amis Groupe S.A.S. - Nouvelle-Aquitaine",
            "fractales amis groupe s a s nouvelle aquitaine",
        ),
        # 7. French with apostrophe and accents
        ("L'Étoile du Nord SARL", "l'étoile du nord sarl"),
        # 8. Hindi with hyphens, brackets, and Devanagari script
        (
            "राम--मार्केटिंग   (प्राइवेट)  लिमिटेड",
            "राम मार्केटिंग प्राइवेट लिमिटेड",
        ),
        # 9. Kannada with extra whitespace and native script
        ("  ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್   ಬೆಂಗಳೂರು  ", "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ ಬೆಂಗಳೂರು"),
        # 10. Telugu with zero-width non-joiner (ZWNJ) and parentheses
        ("తెలుగు ఎంటర్‌ప్రైజెస్ (ఇండియా)", "తెలుగు ఎంటర్ప్రైజెస్ ఇండియా"),
        # 11. Edge cases: None, NaN, and empty strings
        (None, ""),
        (float("nan"), ""),
        ("", ""),
        ("   ", ""),
    ]

    print("=" * 70)
    print("TESTING normalize_name() ACROSS MULTILINGUAL SAMPLES")
    print("=" * 70)

    all_passed = True
    for idx, (raw_input, expected) in enumerate(test_cases, start=1):
        output = normalize_name(raw_input)
        status = "PASS" if output == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] Test #{idx:02d}:")
        print(f"       Input:    {repr(raw_input)}")
        print(f"       Output:   {repr(output)}")
        print(f"       Expected: {repr(expected)}")
        print("-" * 70)

    print(f"\nFinal Result: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}\n")
