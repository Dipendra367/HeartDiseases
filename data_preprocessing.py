import pandas as pd

# Load the original dataset
df = pd.read_csv("npl2019.csv")

# Drop columns with more than 50% missing values
threshold = 0.5
missing_ratio = df.isnull().sum() / len(df)
columns_to_drop = missing_ratio[missing_ratio > threshold].index
df.drop(columns=columns_to_drop, inplace=True)

# Fill numerical columns with mean
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode 'sex' column (if present) - Male = 1, Female = 0
if 'sex' in df.columns:
    df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0})

# ✅ Convert heart disease label from 2 → 0 (No disease), 1 → 1 (Disease)
if 'h17' in df.columns:
    df['h17'] = df['h17'].replace({2: 0})

# Save the cleaned dataset
df.to_csv("cleaned_npl2019.csv", index=False)

# Show final info
print("✅ Cleaned dataset saved as 'cleaned_npl2019.csv'")
print("🎯 Heart disease distribution:\n", df['h17'].value_counts())
