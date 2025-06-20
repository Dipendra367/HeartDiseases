import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_npl2019.csv")

# 📋 Dataset info
print("📋 Dataset Info:")
print(df.info())

# 📈 Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# 💖 Class distribution for h17
sns.countplot(x='h17', data=df)
plt.title("Heart Disease Class Distribution (h17)")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🔍 Correlation with h17
corr = df.corr(numeric_only=True)['h17'].sort_values(ascending=False)
print("\nTop Correlated Features:\n", corr.head(10))
print("\nLeast Correlated Features:\n", corr.tail(10))

# 🔥 Correlation heatmap of top features
top_corr_features = corr[1:11].index  # excluding h17 itself
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top 10 Features Correlated with Heart Disease (h17)")
plt.tight_layout()
plt.show()

# 📊 Age vs Heart Disease
if 'age' in df.columns:
    sns.boxplot(x='h17', y='age', data=df)
    plt.title("Age vs Heart Disease")
    plt.tight_layout()
    plt.show()

# 📊 Cholesterol vs Heart Disease (b14)
if 'b14' in df.columns:
    sns.boxplot(x='h17', y='b14', data=df)
    plt.title("Cholesterol (b14) vs Heart Disease")
    plt.tight_layout()
    plt.show()
