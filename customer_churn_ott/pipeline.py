# pipeline.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==============================
# Paths
# ==============================
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Input files
users_file = DATA_DIR / "users.csv"
watch_file = DATA_DIR / "watch_history.csv"
reviews_file = DATA_DIR / "reviews.csv"
search_file = DATA_DIR / "search_logs.csv"
recommend_file = DATA_DIR / "recommendation_logs.csv"

# Output files
merged_file = DATA_DIR / "netflix_merged.csv"
merged_churn_file = DATA_DIR / "netflix_merged_with_churn.csv"
pred_file = DATA_DIR / "netflix_predictions_with_actuals.csv"
top_file = DATA_DIR / "top_churn_users.csv"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.pkl"

# ==============================
# Step 1: Merge data
# ==============================
print("🔄 Merging data...")

users = pd.read_csv(users_file)
watch = pd.read_csv(watch_file)
reviews = pd.read_csv(reviews_file)
search = pd.read_csv(search_file)
recomm = pd.read_csv(recommend_file)

# Watch features
watch_agg = watch.groupby('user_id').agg(
    total_watch_minutes=('watch_duration_minutes','sum'),
    avg_watch_minutes=('watch_duration_minutes','mean'),
    total_sessions=('session_id','count'),
    unique_genres=('movie_id','nunique')
).reset_index()

# Reviews features
reviews_agg = reviews.groupby('user_id').agg(
    avg_rating=('rating','mean'),
    total_reviews=('review_id','count')
).reset_index()

# Search features
search_agg = search.groupby('user_id').agg(
    total_searches=('search_id','count'),
    unique_search_terms=('search_query','nunique')
).reset_index()

# Recommendation features
recomm_agg = recomm.groupby('user_id').agg(
    total_recommendations=('recommendation_id','count')
).reset_index()

# Merge all features
df = users.merge(watch_agg, on='user_id', how='left') \
          .merge(reviews_agg, on='user_id', how='left') \
          .merge(search_agg, on='user_id', how='left') \
          .merge(recomm_agg, on='user_id', how='left')

df.fillna(0, inplace=True)
df.to_csv(merged_file, index=False)
print(f"✅ Data merged: {merged_file}, shape: {df.shape}")

# ==============================
# Step 2: Add churn label
# ==============================
# Churn if avg_watch_minutes < 1 OR total_sessions < 5
df['churned'] = ((df['avg_watch_minutes'] < 1) | (df['total_sessions'] < 5)).astype(int)
df.to_csv(merged_churn_file, index=False)
print(f"✅ Churn label added: {merged_churn_file}")

# ==============================
# Step 3: Train/Test split
# ==============================
X = df.drop(columns=['user_id','email','first_name','last_name','churned'], errors='ignore')
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Step 4: Encode categorical
# ==============================
encoders = {}
for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, ENCODERS_PATH)
print(f"✅ Encoders saved: {ENCODERS_PATH}")

# ==============================
# Step 5: Scale numeric features
# ==============================
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=['int64','float64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved: {SCALER_PATH}")

# ==============================
# Step 6: Handle imbalance with SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_res.shape}, {y_train_res.shape}")

# ==============================
# Step 7: Train model
# ==============================
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {model.score(X_test, y_test):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and feature columns
joblib.dump(model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURE_COLS_PATH)
print(f"✅ Model saved: {MODEL_PATH}")
print(f"✅ Feature columns saved: {FEATURE_COLS_PATH}")

# ==============================
# Step 8: Predict on full dataset
# ==============================
# Apply encoders and scaler on full X
for col, le in encoders.items():
    if col in X.columns:
        X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X[col] = le.transform(X[col].astype(str))

X[num_cols] = scaler.transform(X[num_cols])

churn_pred = model.predict(X)
churn_prob = model.predict_proba(X)[:,1]

df_output = df[['user_id','churned']].copy()
df_output.rename(columns={'churned':'churned_actual'}, inplace=True)
df_output['churn_prediction'] = churn_pred
df_output['churn_probability'] = churn_prob
df_output.to_csv(pred_file, index=False)
print(f"✅ Predictions saved: {pred_file}")

# ==============================
# Step 9: Top 100 likely churn users
# ==============================
top_churn = df_output.sort_values(by='churn_probability', ascending=False).head(100)
top_churn.to_csv(top_file, index=False)
print(f"✅ Top 100 likely churn users saved: {top_file}")
