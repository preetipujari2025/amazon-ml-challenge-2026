import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import re
import unittest
from rapidfuzz import fuzz

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()

def extract_pin(text):
    text = safe_str(text)
    matches = re.findall(r'\b\d{5,6}\b', text)
    return matches[-1] if matches else None

def categorize_false_positive(row):
    """
    row must contain:
    business_name_source1, business_name_candidate
    business_address_source1, business_address_candidate
    country_source1, country_candidate
    """
    n1 = safe_str(row.get('business_name_source1', '')).lower()
    n2 = safe_str(row.get('business_name_candidate', '')).lower()
    a1 = safe_str(row.get('business_address_source1', '')).lower()
    a2 = safe_str(row.get('business_address_candidate', '')).lower()
    c1 = safe_str(row.get('country_source1', '')).lower()
    c2 = safe_str(row.get('country_candidate', '')).lower()
    
    name_sim = fuzz.ratio(n1, n2)
    addr_sim = fuzz.ratio(a1, a2)
    
    if c1 and c2 and c1 != c2:
        return 'country_mismatch'
        
    if name_sim > 80:
        return 'similar_name'
        
    if a1 and a2 and addr_sim > 90:
        return 'same_address'
        
    if n1 and n2 and (len(n1) < 5 or len(n2) < 5):
        return 'common_or_short_name'
        
    return 'other'
    
def categorize_false_negative(row):
    n1 = safe_str(row.get('business_name_source1', '')).lower()
    n2 = safe_str(row.get('business_name_candidate', '')).lower()
    a1 = safe_str(row.get('business_address_source1', '')).lower()
    a2 = safe_str(row.get('business_address_candidate', '')).lower()
    
    if not n1 or not n2 or not a1 or not a2:
        return 'missing_field'
        
    # Check for abbreviation (e.g., inc vs incorporated)
    n1_words = set(n1.split())
    n2_words = set(n2.split())
    if len(n1_words) != len(n2_words) and (n1_words.issubset(n2_words) or n2_words.issubset(n1_words)):
        return 'abbreviation'
        
    # Check for spelling variation (high token sort ratio but low exact ratio)
    if fuzz.token_sort_ratio(n1, n2) > 85 and fuzz.ratio(n1, n2) < 85:
        return 'spelling_variation'
        
    pin1 = extract_pin(a1)
    pin2 = extract_pin(a2)
    if (not pin1 and pin2) or (pin1 and not pin2):
        return 'missing_pin'
        
    # Partial address
    if (a1 in a2 or a2 in a1) and abs(len(a1) - len(a2)) > 5:
        return 'partial_address'
        
    # Assume transliteration or other
    if fuzz.ratio(n1, n2) > 70:
        return 'transliteration'
        
    return 'other'

def run_error_analysis(predictions_df, s1_df, cand_df, out_txt="experiments/error_analysis.txt", out_tsv="experiments/error_cases.tsv"):
    """
    predictions_df must contain: source1_entity_id, candidate_entity_id, score, prediction, ground_truth_label
    """
    req_cols = ['source1_entity_id', 'candidate_entity_id', 'score', 'prediction', 'ground_truth_label']
    for c in req_cols:
        if c not in predictions_df.columns:
            raise ValueError(f"Missing column {c} in predictions dataframe.")
            
    # Merge with original data
    df = predictions_df.copy()
    
    s1_subset = s1_df[['entity_id', 'business_name', 'business_address', 'country']].rename(columns={
        'entity_id': 'source1_entity_id',
        'business_name': 'business_name_source1',
        'business_address': 'business_address_source1',
        'country': 'country_source1'
    })
    
    cand_subset = cand_df[['entity_id', 'business_name', 'business_address', 'country']].rename(columns={
        'entity_id': 'candidate_entity_id',
        'business_name': 'business_name_candidate',
        'business_address': 'business_address_candidate',
        'country': 'country_candidate'
    })
    
    df = df.merge(s1_subset, on='source1_entity_id', how='left')
    df = df.merge(cand_subset, on='candidate_entity_id', how='left')
    
    # Identify error types
    df['error_type'] = 'correct'
    df.loc[(df['prediction'] == 1) & (df['ground_truth_label'] == 0), 'error_type'] = 'FP'
    df.loc[(df['prediction'] == 0) & (df['ground_truth_label'] == 1), 'error_type'] = 'FN'
    
    df['category'] = 'N/A'
    
    # Categorize errors
    fp_mask = df['error_type'] == 'FP'
    if fp_mask.any():
        df.loc[fp_mask, 'category'] = df[fp_mask].apply(categorize_false_positive, axis=1)
        
    fn_mask = df['error_type'] == 'FN'
    if fn_mask.any():
        df.loc[fn_mask, 'category'] = df[fn_mask].apply(categorize_false_negative, axis=1)
        
    # Calculate similarities for diagnostics
    def calc_sims(row):
        n1 = safe_str(row.get('business_name_source1', ''))
        n2 = safe_str(row.get('business_name_candidate', ''))
        a1 = safe_str(row.get('business_address_source1', ''))
        a2 = safe_str(row.get('business_address_candidate', ''))
        
        return pd.Series({
            'name_similarity': fuzz.ratio(n1.lower(), n2.lower()),
            'address_similarity': fuzz.ratio(a1.lower(), a2.lower()),
            'country_match': 1 if safe_str(row.get('country_source1')) == safe_str(row.get('country_candidate')) and safe_str(row.get('country_source1')) else 0
        })
        
    df[['name_similarity', 'address_similarity', 'country_match']] = df.apply(calc_sims, axis=1)
    
    # Save TSV
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    df[df['error_type'] != 'correct'].to_csv(out_tsv, sep='\t', index=False)
    
    # Generate Summary
    total = len(df)
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    fp_rate = fp_count / total if total > 0 else 0
    fn_rate = fn_count / total if total > 0 else 0
    
    fp_cats = df[fp_mask]['category'].value_counts().to_dict()
    fn_cats = df[fn_mask]['category'].value_counts().to_dict()
    
    summary = []
    summary.append("=== ERROR ANALYSIS SUMMARY ===")
    summary.append(f"Total validation pairs: {total}")
    summary.append(f"False positives: {fp_count}")
    summary.append(f"False negatives: {fn_count}")
    summary.append(f"False-positive rate: {fp_rate:.4f}")
    summary.append(f"False-negative rate: {fn_rate:.4f}")
    
    summary.append("\nMost common false-positive patterns:")
    for cat, cnt in fp_cats.items():
        summary.append(f" - {cat}: {cnt}")
        
    summary.append("\nMost common false-negative patterns:")
    for cat, cnt in fn_cats.items():
        summary.append(f" - {cat}: {cnt}")
        
    summary.append("\nExamples of important errors:")
    err_examples = df[df['error_type'] != 'correct'].head(5)
    for _, row in err_examples.iterrows():
        summary.append(f"[{row['error_type']}] S1: {row['source1_entity_id']} | Cand: {row['candidate_entity_id']} | Score: {row['score']:.4f} | Cat: {row['category']}")
        summary.append(f"   Name1: {row.get('business_name_source1')} | Name2: {row.get('business_name_candidate')}")
        
    summary_text = "\n".join(summary)
    
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        f.write(summary_text)
        
    return summary_text


class TestErrorAnalysis(unittest.TestCase):
    def test_fp_category_country_mismatch(self):
        row = {
            'business_name_source1': 'Acme', 'business_name_candidate': 'Acme',
            'country_source1': 'US', 'country_candidate': 'UK'
        }
        self.assertEqual(categorize_false_positive(row), 'country_mismatch')
        
    def test_fp_category_similar_name(self):
        row = {
            'business_name_source1': 'Acme Corp', 'business_name_candidate': 'Acme Corp Inc',
            'country_source1': 'US', 'country_candidate': 'US'
        }
        self.assertEqual(categorize_false_positive(row), 'similar_name')
        
    def test_fn_category_missing_field(self):
        row = {
            'business_name_source1': 'Acme Corp', 'business_name_candidate': '',
        }
        self.assertEqual(categorize_false_negative(row), 'missing_field')
        
    def test_fn_category_missing_pin(self):
        row = {
            'business_name_source1': 'Acme Corp', 'business_name_candidate': 'Acme Corp',
            'business_address_source1': '123 Main St 90210', 'business_address_candidate': '123 Main St',
            'country_source1': 'US', 'country_candidate': 'US'
        }
        self.assertEqual(categorize_false_negative(row), 'missing_pin')

if __name__ == '__main__':
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="Run Error Analysis on Entity Resolution Predictions")
    parser.add_argument("--predictions", type=str, default="models/val_predictions_logistic_regression.tsv",
                        help="Path to validation predictions TSV.")
    parser.add_argument("--threshold_json", type=str, default="models/threshold.json",
                        help="Path to optimal threshold JSON.")
    parser.add_argument("--gt_data", type=str, default="dataset/train/train_ground_truth.tsv",
                        help="Path to ground truth TSV.")
    parser.add_argument("--s1_data", type=str, default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2_data", type=str, default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3_data", type=str, default="dataset/train/train_source3.tsv")
    parser.add_argument("--out_txt", type=str, default="experiments/error_analysis.txt",
                        help="Output path for summary report.")
    parser.add_argument("--out_tsv", type=str, default="experiments/error_cases.tsv",
                        help="Output path for detailed error cases.")
    parser.add_argument("--test", action="store_true", help="Run unit tests instead of error analysis.")
    args = parser.parse_args()

    if args.test:
        print("Running Error Analysis Unit Tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestErrorAnalysis)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
    else:
        if not os.path.exists(args.predictions):
            print(f"Predictions file not found: {args.predictions}")
            sys.exit(1)
            
        print(f"Loading predictions from {args.predictions}...")
        pred_df = pd.read_csv(args.predictions, sep='\t')
        
        # Determine threshold
        threshold = 0.5
        if os.path.exists(args.threshold_json):
            with open(args.threshold_json, 'r') as f:
                th_data = json.load(f)
                threshold = th_data.get('threshold', 0.5)
            print(f"Using threshold {threshold:.2f} from {args.threshold_json}")
        else:
            print(f"Threshold file not found; defaulting to {threshold:.2f}")
            
        pred_df['prediction'] = (pred_df['score'] >= threshold).astype(int)
        
        # Load ground truth
        print(f"Loading ground truth from {args.gt_data}...")
        gt_df = pd.read_csv(args.gt_data, sep='\t', dtype=str)
        gt_map = {}
        for s1, m in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
            if pd.notna(m) and m.strip():
                gt_map[s1] = set(m.strip().split(','))
            else:
                gt_map[s1] = set()
                
        pred_df['ground_truth_label'] = [
            1 if cand in gt_map.get(s1, set()) else 0
            for s1, cand in zip(pred_df['source1_entity_id'], pred_df['candidate_entity_id'])
        ]
        
        # Load metadata for involved entities
        needed_s1 = set(pred_df['source1_entity_id'])
        needed_cand = set(pred_df['candidate_entity_id'])
        
        print("Loading metadata for error analysis entities...")
        s1_df = pd.read_csv(args.s1_data, sep='\t')
        s1_df = s1_df[s1_df['entity_id'].isin(needed_s1)].reset_index(drop=True)
        
        cand_list = []
        for p in [args.s2_data, args.s3_data]:
            df_c = pd.read_csv(p, sep='\t')
            df_c = df_c[df_c['entity_id'].isin(needed_cand)]
            cand_list.append(df_c)
        cand_df = pd.concat(cand_list, ignore_index=True)
        
        print("Running error analysis...")
        summary = run_error_analysis(pred_df, s1_df, cand_df, out_txt=args.out_txt, out_tsv=args.out_tsv)
        print("\n" + summary)
        print(f"\nError analysis completed. Outputs: {args.out_txt}, {args.out_tsv}")

