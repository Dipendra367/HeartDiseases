import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load the cleaned dataset
df = pd.read_csv("cleaned_npl2019.csv")

# ✅ Feature list (based on EDA + domain knowledge)
features = ['age', 'sex', 'b14', 'b15', 'b16', 'dx4', 'o3', 'c7', 'p6b', 'b10']
X = df[features]
y = df['h17']

# 🔀 Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔄 Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚖️ Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# 🧠 Train XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 💾 Save the model, scaler, and test data
pickle.dump(model, open("xgb_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump((X_test_scaled, y_test), open("test_data.pkl", "wb"))

print("✅ Model trained and saved as 'xgb_model.pkl'")
print("📦 Scaler and test data saved as 'scaler.pkl' and 'test_data.pkl'")
