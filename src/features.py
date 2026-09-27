import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer
# Note: cosine similarity is computed directly via sparse matrix dot product
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureGenerator:
    """
    Generates pairwise features for business entity resolution.
    Supports fit, transform, and fit_transform methods to ensure reusable TF-IDF vectorizers.
    """
    def __init__(self, tfidf_ngram_range=(3, 5), tfidf_max_features=10000):
        self.name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=tfidf_ngram_range, max_features=tfidf_max_features, lowercase=True)
        self.address_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=tfidf_ngram_range, max_features=tfidf_max_features, lowercase=True)
        self.is_fitted = False
        
        self.feature_names = [
            'name_norm_levenshtein', 'name_rapidfuzz_levenshtein', 'name_jaro_winkler', 
            'name_token_similarity', 'name_token_set_similarity', 'name_token_sort_similarity', 
            'name_token_jaccard', 'name_tfidf_cosine', 'name_exact_norm_match', 'name_exact_core_match',
            'name_len_diff', 'name_norm_len_diff',
            'addr_norm_levenshtein', 'addr_token_similarity', 'addr_token_set_similarity', 
            'addr_token_sort_similarity', 'addr_token_jaccard', 'addr_tfidf_cosine', 
            'addr_exact_norm_match', 'addr_pin_exact_match', 'addr_pin_missing',
            'addr_len_diff', 'addr_norm_len_diff',
            'country_exact_match', 'country_missing_s1', 'country_missing_cand',
            'name_missing_s1', 'name_missing_cand', 'addr_missing_s1', 'addr_missing_cand',
            'source_pair_indicator'
        ]

    def _safe_str(self, val):
        if pd.isna(val) or val is None:
            return ""
        return str(val).strip()

    def _normalize_string(self, text):
        text = self._safe_str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_core_name(self, text):
        # A simple placeholder for extracting core name, e.g., removing common suffixes like inc, llc
        text = self._normalize_string(text)
        text = re.sub(r'\b(inc|llc|ltd|company|corporation|corp)\b', '', text).strip()
        return text

    def _extract_pin(self, text):
        # Simple extraction for 5 or 6 digit PIN codes
        matches = re.findall(r'\b\d{5,6}\b', text)
        return matches[-1] if matches else None
        
    def _jaccard_similarity(self, str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def fit(self, df_s1, df_cand, pairs):
        """
        Fits the TF-IDF vectorizers on the provided data.
        df_s1: DataFrame of Source 1 records
        df_cand: DataFrame of Candidate records (Source 2/3)
        pairs: DataFrame with columns ['source1_entity_id', 'candidate_entity_id']
        """
        s1_names = df_s1['business_name'].apply(self._normalize_string).tolist()
        cand_names = df_cand['business_name'].apply(self._normalize_string).tolist()
        all_names = s1_names + cand_names
        
        s1_addrs = df_s1['business_address'].apply(self._normalize_string).tolist()
        cand_addrs = df_cand['business_address'].apply(self._normalize_string).tolist()
        all_addrs = s1_addrs + cand_addrs

        # Fit vectorizers
        self.name_vectorizer.fit(all_names)
        self.address_vectorizer.fit(all_addrs)
        self.is_fitted = True
        return self

    def transform(self, df_s1, df_cand, pairs):
        """
        Generates features for the given pairs.
        df_s1: DataFrame or dict of Source 1 records
        df_cand: DataFrame or dict of Candidate records (Source 2/3)
        pairs: DataFrame with columns ['source1_entity_id', 'candidate_entity_id']
               or ['source1_id', 'candidate_id']
        """
        if not self.is_fitted:
            raise ValueError("FeatureGenerator must be fitted before calling transform.")

        s1_col = 'source1_entity_id' if 'source1_entity_id' in pairs.columns else 'source1_id'
        cand_col = 'candidate_entity_id' if 'candidate_entity_id' in pairs.columns else 'candidate_id'
        s1_ids = pairs[s1_col].values
        cand_ids = pairs[cand_col].values

        if isinstance(df_s1, dict):
            s1_name_map = df_s1.get('name', {})
            s1_addr_map = df_s1.get('addr', {})
            s1_ctry_map = df_s1.get('country', {})
        else:
            if 'entity_id' in df_s1.columns:
                df_s1 = df_s1.set_index('entity_id')
            s1_name_map = df_s1['business_name'].to_dict()
            s1_addr_map = df_s1['business_address'].to_dict()
            s1_ctry_map = df_s1['country'].to_dict()

        if isinstance(df_cand, dict):
            cand_name_map = df_cand.get('name', {})
            cand_addr_map = df_cand.get('addr', {})
            cand_ctry_map = df_cand.get('country', {})
        else:
            if 'entity_id' in df_cand.columns:
                df_cand = df_cand.set_index('entity_id')
            cand_name_map = df_cand['business_name'].to_dict()
            cand_addr_map = df_cand['business_address'].to_dict()
            cand_ctry_map = df_cand['country'].to_dict()

        s1_names = [self._safe_str(s1_name_map.get(idx, "")) for idx in s1_ids]
        cand_names = [self._safe_str(cand_name_map.get(idx, "")) for idx in cand_ids]
        s1_addrs = [self._safe_str(s1_addr_map.get(idx, "")) for idx in s1_ids]
        cand_addrs = [self._safe_str(cand_addr_map.get(idx, "")) for idx in cand_ids]
        s1_countries = [self._safe_str(s1_ctry_map.get(idx, "")) for idx in s1_ids]
        cand_countries = [self._safe_str(cand_ctry_map.get(idx, "")) for idx in cand_ids]
        
        # Normalize
        s1_names_norm = [self._normalize_string(n) for n in s1_names]
        cand_names_norm = [self._normalize_string(n) for n in cand_names]
        
        s1_addrs_norm = [self._normalize_string(a) for a in s1_addrs]
        cand_addrs_norm = [self._normalize_string(a) for a in cand_addrs]

        # TF-IDF calculations
        name_tfidf_s1 = self.name_vectorizer.transform(s1_names_norm)
        name_tfidf_cand = self.name_vectorizer.transform(cand_names_norm)
        name_cosine_sims = np.asarray(name_tfidf_s1.multiply(name_tfidf_cand).sum(axis=1)).flatten()
        
        addr_tfidf_s1 = self.address_vectorizer.transform(s1_addrs_norm)
        addr_tfidf_cand = self.address_vectorizer.transform(cand_addrs_norm)
        addr_cosine_sims = np.asarray(addr_tfidf_s1.multiply(addr_tfidf_cand).sum(axis=1)).flatten()

        features = []
        for i in range(len(pairs)):
            row_features = []
            
            s1_name_raw, cand_name_raw = s1_names[i], cand_names[i]
            s1_name, cand_name = s1_names_norm[i], cand_names_norm[i]
            
            s1_addr_raw, cand_addr_raw = s1_addrs[i], cand_addrs[i]
            s1_addr, cand_addr = s1_addrs_norm[i], cand_addrs_norm[i]
            
            s1_country, cand_country = s1_countries[i].lower(), cand_countries[i].lower()
            
            # --- NAME FEATURES ---
            if s1_name and cand_name:
                row_features.extend([
                    distance.Levenshtein.normalized_similarity(s1_name, cand_name),
                    fuzz.ratio(s1_name, cand_name) / 100.0,
                    distance.JaroWinkler.similarity(s1_name, cand_name),
                    fuzz.ratio(s1_name, cand_name) / 100.0, # token similarity proxy
                    fuzz.token_set_ratio(s1_name, cand_name) / 100.0,
                    fuzz.token_sort_ratio(s1_name, cand_name) / 100.0,
                    self._jaccard_similarity(s1_name, cand_name),
                    name_cosine_sims[i],
                    1.0 if s1_name == cand_name else 0.0,
                    1.0 if self._extract_core_name(s1_name) == self._extract_core_name(cand_name) and self._extract_core_name(s1_name) != "" else 0.0,
                    abs(len(s1_name_raw) - len(cand_name_raw)),
                    abs(len(s1_name) - len(cand_name))
                ])
            else:
                row_features.extend([0.0] * 12)
                
            # --- ADDRESS FEATURES ---
            if s1_addr and cand_addr:
                s1_pin = self._extract_pin(s1_addr_raw)
                cand_pin = self._extract_pin(cand_addr_raw)
                pin_exact = 1.0 if (s1_pin and cand_pin and s1_pin == cand_pin) else 0.0
                pin_missing = 1.0 if (not s1_pin or not cand_pin) else 0.0
                
                row_features.extend([
                    distance.Levenshtein.normalized_similarity(s1_addr, cand_addr),
                    fuzz.ratio(s1_addr, cand_addr) / 100.0, # token similarity proxy
                    fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0,
                    fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0,
                    self._jaccard_similarity(s1_addr, cand_addr),
                    addr_cosine_sims[i],
                    1.0 if s1_addr == cand_addr else 0.0,
                    pin_exact,
                    pin_missing,
                    abs(len(s1_addr_raw) - len(cand_addr_raw)),
                    abs(len(s1_addr) - len(cand_addr))
                ])
            else:
                row_features.extend([0.0] * 11)
                
            # --- OTHER FEATURES ---
            cand_id = cand_ids[i]
            source_pair_indicator = 0.0
            if "S2" in cand_id or str(cand_id).startswith("2"): 
                source_pair_indicator = 1.0 # S1-S2
            elif "S3" in cand_id or str(cand_id).startswith("3"): 
                source_pair_indicator = 2.0 # S1-S3
                
            row_features.extend([
                1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0,
                1.0 if not s1_country else 0.0,
                1.0 if not cand_country else 0.0,
                1.0 if not s1_name_raw else 0.0,
                1.0 if not cand_name_raw else 0.0,
                1.0 if not s1_addr_raw else 0.0,
                1.0 if not cand_addr_raw else 0.0,
                source_pair_indicator
            ])
            
            features.append(row_features)

        # Convert to DataFrame
        feat_df = pd.DataFrame(features, columns=self.feature_names)
        
        # Ensure no NaNs or Infs
        feat_df = feat_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        # Prepend IDs
        result_df = pd.DataFrame({
            'source1_entity_id': s1_ids,
            'candidate_entity_id': cand_ids
        })
        result_df = pd.concat([result_df, feat_df], axis=1)
        
        return result_df

    def fit_transform(self, df_s1, df_cand, pairs):
        self.fit(df_s1, df_cand, pairs)
        return self.transform(df_s1, df_cand, pairs)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)

if __name__ == "__main__":
    # Small feature generation test
    print("Running feature generation test...")
    df_s1 = pd.DataFrame({
        'entity_id': ['S1_1', 'S1_2'],
        'business_name': ['Amazon Inc', 'Google LLC'],
        'business_address': ['123 Amazon Way, Seattle WA 98109', '1600 Amphitheatre Pkwy, Mountain View CA 94043'],
        'country': ['US', 'US']
    })
    
    df_cand = pd.DataFrame({
        'entity_id': ['S2_1', 'S3_1'],
        'business_name': ['Amazon.com Inc.', 'Alphabet (Google)'],
        'business_address': ['123 Amazon Way, Seattle 98109', '1600 Amphitheatre, CA'],
        'country': ['US', 'USA']
    })
    
    pairs = pd.DataFrame({
        'source1_entity_id': ['S1_1', 'S1_2'],
        'candidate_entity_id': ['S2_1', 'S3_1']
    })
    
    fg = FeatureGenerator()
    feat_df = fg.fit_transform(df_s1, df_cand, pairs)
    
    print(f"Number of candidate pairs processed: {len(feat_df)}")
    print(f"Number of feature columns: {len(fg.feature_names)}")
    print(f"Feature names: {fg.feature_names}")
    
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    num_nans = feat_df[numeric_cols].isna().sum().sum()
    num_infs = np.isinf(feat_df[numeric_cols]).sum().sum()
    
    print(f"Number of NaN values: {num_nans}")
    print(f"Number of infinite values: {num_infs}")
    
    if num_nans > 0 or num_infs > 0:
        raise ValueError("Features contain NaN or infinite values!")
    
    print("Test passed successfully!")
