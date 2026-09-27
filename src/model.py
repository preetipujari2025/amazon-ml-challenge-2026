import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.special import expit

# Fallback Scipy-based Logistic Regression
class ScipyLogisticRegression:
    def __init__(self, class_weight='balanced', max_iter=1000, random_state=42, C=1.0):
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.C = C
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_mat.shape

        if self.class_weight == 'balanced':
            n_pos = np.sum(y_vec == 1)
            n_neg = np.sum(y_vec == 0)
            w_pos = n_samples / (2.0 * max(n_pos, 1))
            w_neg = n_samples / (2.0 * max(n_neg, 1))
            sample_weights = np.where(y_vec == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(n_samples)

        self.mean_ = np.mean(X_mat, axis=0)
        self.std_ = np.std(X_mat, axis=0)
        self.std_[self.std_ == 0] = 1.0
        X_norm = (X_mat - self.mean_) / self.std_

        X_b = np.hstack([np.ones((n_samples, 1)), X_norm])
        w0 = np.zeros(n_features + 1)
        reg = 1.0 / max(self.C, 1e-4)

        def loss_and_grad(w):
            z = np.clip(X_b @ w, -30, 30)
            p = expit(z)
            eps = 1e-15
            loss = -np.sum(sample_weights * (y_vec * np.log(p + eps) + (1 - y_vec) * np.log(1 - p + eps))) / n_samples
            loss += 0.5 * reg * np.sum(w[1:] ** 2) / n_samples
            grad = (X_b.T @ (sample_weights * (p - y_vec))) / n_samples
            grad[1:] += reg * w[1:] / n_samples
            return loss, grad

        res = minimize(loss_and_grad, w0, jac=True, method='L-BFGS-B', options={'maxiter': self.max_iter})
        self.intercept_ = np.array([res.x[0]])
        self.coef_ = np.array([res.x[1:] / self.std_])
        self.intercept_ -= np.sum(self.coef_ * self.mean_)
        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=np.float64)
        z = np.clip(X_mat @ self.coef_.T + self.intercept_, -30, 30)
        p1 = expit(z).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

# Fallback Random Forest
class SimpleTree:
    def __init__(self, max_depth=6, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _fit_node(self, X, y, depth, max_features, rng):
        n_samples, n_features = X.shape
        p1 = float(np.mean(y)) if n_samples > 0 else 0.0
        if depth >= self.max_depth or n_samples < self.min_samples_split or p1 == 0.0 or p1 == 1.0:
            return {'leaf': True, 'prob': p1}

        k = min(max_features, n_features) if max_features else n_features
        feat_indices = rng.choice(n_features, k, replace=False)
        best_gain = -1.0
        best_split = None

        current_gini = 1.0 - (p1**2 + (1.0 - p1)**2)

        for f in feat_indices:
            col_vals = X[:, f]
            vals = np.unique(col_vals)
            if len(vals) > 10:
                vals = np.percentile(vals, np.linspace(10, 90, 9))
            for thresh in vals:
                left_mask = col_vals <= thresh
                n_l = int(np.sum(left_mask))
                n_r = n_samples - n_l
                if n_l == 0 or n_r == 0:
                    continue
                p_l = float(np.mean(y[left_mask]))
                p_r = float(np.mean(y[~left_mask]))
                gini_l = 1.0 - (p_l**2 + (1.0 - p_l)**2)
                gini_r = 1.0 - (p_r**2 + (1.0 - p_r)**2)
                gain = current_gini - (n_l / n_samples * gini_l + n_r / n_samples * gini_r)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, thresh)

        if best_split is None or best_gain <= 0:
            return {'leaf': True, 'prob': p1}

        f, thresh = best_split
        left_mask = X[:, f] <= thresh
        return {
            'leaf': False,
            'feature': f,
            'threshold': thresh,
            'left': self._fit_node(X[left_mask], y[left_mask], depth + 1, max_features, rng),
            'right': self._fit_node(X[~left_mask], y[~left_mask], depth + 1, max_features, rng)
        }

    def fit(self, X, y, max_features=None, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        self.root = self._fit_node(X, y, 0, max_features, rng)
        return self

    def _predict_row(self, node, x):
        if node['leaf']:
            return node['prob']
        if x[node['feature']] <= node['threshold']:
            return self._predict_row(node['left'], x)
        return self._predict_row(node['right'], x)

    def predict_proba(self, X):
        return np.array([self._predict_row(self.root, row) for row in X])

class SimpleRandomForest:
    def __init__(self, n_estimators=20, max_depth=6, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_mat.shape

        rng = np.random.RandomState(self.random_state)
        mf = int(np.sqrt(n_features)) if self.max_features == 'sqrt' else n_features

        self.trees = []
        for i in range(self.n_estimators):
            sample_indices = rng.choice(n_samples, n_samples, replace=True)
            tree = SimpleTree(max_depth=self.max_depth)
            tree.fit(X_mat[sample_indices], y_vec[sample_indices], max_features=mf, rng=rng)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X_mat = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict_proba(X_mat) for tree in self.trees])
        p1 = np.mean(all_preds, axis=0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class EntityResolutionModel:
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        
        # Try importing scikit-learn; fall back to Scipy/NumPy if C-extensions are blocked
        if self.model_type == 'logistic_regression':
            try:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    random_state=self.random_state, 
                    class_weight='balanced',
                    max_iter=1000
                )
            except Exception:
                self.model = ScipyLogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_iter=1000
                )
        elif self.model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_estimators=100,
                    n_jobs=-1
                )
            except Exception:
                self.model = SimpleRandomForest(
                    n_estimators=20,
                    max_depth=6,
                    random_state=self.random_state
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        """
        Trains the model on the provided features and labels.
        """
        if len(y.unique()) <= 1:
            raise ValueError("Training data must contain more than one class.")
            
        # Ensure only numeric features are used
        non_numeric = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Features contain non-numeric columns: {non_numeric.tolist()}")
            
        if X.isna().any().any():
            raise ValueError("Features contain NaN values.")
            
        if (X == float('inf')).any().any() or (X == float('-inf')).any().any():
            raise ValueError("Features contain infinite values.")
            
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns probability of matching (class 1).
        """
        return self.model.predict_proba(X)
        
    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath, model_type):
        instance = cls(model_type=model_type)
        instance.model = joblib.load(filepath)
        return instance
