# train_netflix_churn_model.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Paths
# ==============================
DATA_PATH = Path("data/netflix_merged_with_churn.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "netflix_churn_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "netflix_feature_columns.pkl"
ENCODERS_PATH = MODEL_DIR / "netflix_encoders.pkl"

# ==============================
# Load Data
# ==============================
df = pd.read_csv(DATA_PATH)

if "churned" not in df.columns:
    raise ValueError("❌ Target column 'churned' not found in dataset!")

target = "churned"
X = df.drop(columns=['user_id','email','first_name','last_name','churned'])
y = df[target]

# ==============================
# Encode Categorical Columns
# ==============================
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, ENCODERS_PATH)
print(f"✅ Encoders saved to {ENCODERS_PATH}")

# ==============================
# Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")

# ==============================
# Train Model
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ==============================
# Evaluate Model
# ==============================
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# Save Model & Features
# ==============================
joblib.dump(model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURE_COLS_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Feature columns saved to {FEATURE_COLS_PATH}")
