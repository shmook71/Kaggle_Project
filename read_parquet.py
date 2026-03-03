import pandas as pd

df = pd.read_parquet("flights_RUH.parquet")

print("✅ عدد الصفوف والأعمدة:", df.shape)
print("\n✅ أول 5 صفوف:")
print(df.head())
print("\n✅ أسماء الأعمدة:")
print(df.columns)