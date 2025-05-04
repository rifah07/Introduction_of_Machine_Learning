import pandas as pd

df = pd.read_excel("salary_march_25.xlsx")
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# 1 Count how many employees have the same salary
salary_counts = df['Salary'].value_counts()
print("Employees with equal salaries:")
print(salary_counts)


#2
engineering_count = df['Department'].str.contains("Engineering", case=False, na=False).sum()
print(f"Number of employees in Engineering department: {engineering_count}")

#3
probationary_count = df['Job Status'].str.contains("Probationary", case=False, na=False).sum()
print(f"Number of probationary officers: {probationary_count}")
