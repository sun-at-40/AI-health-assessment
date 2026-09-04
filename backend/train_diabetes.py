import os
import pickle

# --- Configuration ---
# Robust path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "diabetes.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

def train_diabetes_model():
    print("Starting Diabetes Model Training...")
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    try:
        import xgboost as xgb
    except ImportError:
        raise RuntimeError("xgboost is required to train the diabetes model. Install it or mock it in tests.")

    # 1. Load Data
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded Dataset: {len(df)} records")

    # 2. Select the same BRFSS features sent by the prediction endpoint.
    feature_cols = [
        'HighBP', 'HighChol', 'BMI', 'Smoker', 'HeartDiseaseorAttack',
        'PhysActivity', 'GenHlth', 'Sex', 'Age'
    ]
    X = df[feature_cols]
    Y = df['diabetes'].astype(int)

    # 4. Train/Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 5. Training (XGBoost)
    # Note: No scaling is performed, matching the notebook and ml_service behavior.
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, Y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    print(f"Model Trained. Accuracy: {acc:.4f}")

    # 7. Save Model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model Saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_diabetes_model()
