import pandas as pd
import os

df = pd.read_excel("salary_march_25.xlsx")
df.to_csv("salary_march_temp.csv", index=False)
print(f"Temporary CSV created with all rows")
print(df)


df_without_title = pd.read_csv("salary_march_temp.csv", skiprows=1)
df_without_title.to_csv("salary_march.csv", index=False)
print(f"Final CSV created with first row removed")
print(df_without_title)