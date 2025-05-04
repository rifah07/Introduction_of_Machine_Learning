import pandas as pd
import numpy as np


df= pd.read_csv("salary_march.csv")


#1
salary_counts = df['Salary'].value_counts()
print("Employees with equal salaries:")
print(salary_counts)


#2
engineering_count = df['Department'].str.contains("Engineering", case=False, na=False).sum()
print(f"Number of employees in Engineering department: {engineering_count}")

#3
probationary_count = df['Job Status'].str.contains("Probation", case=False, na=False).sum()
print(f"Number of probationary officers: {probationary_count}")
