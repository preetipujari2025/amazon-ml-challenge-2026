from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

class EntityResolutionModel:
    def __init__(self, model_type='logistic_regression', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=self.random_state, 
                class_weight='balanced',
                max_iter=1000
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_estimators=100,
                n_jobs=-1
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
