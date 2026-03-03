import pandas as pd

df = pd.read_parquet("flights_RUH.parquet")
df.to_csv("flights_RUH.csv", index=False)

print("تم التحويل إلى CSV ✅")