import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 🔄 Load trained model, scaler, and test data
model = pickle.load(open("xgb_model.pkl", "rb"))
X_test, y_test = pickle.load(open("test_data.pkl", "rb"))

# ✅ Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 📊 Evaluation Metrics
print("✅ Model Evaluation Results\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
