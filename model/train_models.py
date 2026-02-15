import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# Set project paths FIRST
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(os.path.join(BASE_DIR, "adult.csv"))

# ===============================
# Encode categorical columns
# ===============================
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("income", axis=1)
y = df["income"]

# ===============================
# Scaling
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ===============================
# Save preprocessing objects
# ===============================
joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ===============================
# Models
# ===============================
models = {
    "logistic_model.pkl": LogisticRegression(max_iter=1000),
    "dt_model.pkl": DecisionTreeClassifier(),
    "knn_model.pkl": KNeighborsClassifier(),
    "nb_model.pkl": GaussianNB(),
    "rf_model.pkl": RandomForestClassifier(),
    "xgb_model.pkl": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, name))

print("All models, encoders, and scaler saved successfully.")