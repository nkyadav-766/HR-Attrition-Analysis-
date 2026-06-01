import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  

df=pd.read_csv("HR_Attrition_Raw.csv")

df = df[df['EmployeeID'].notna()]
df["EmployeeID"].isnull().sum()

df["Age"] = df["Age"].replace({
    -5: 5,
    48.33094995: 48,
    999: 99
})

df["Age"]=df["Age"].fillna(df["Age"].mean())

df["Department"]=df["Department"].fillna("other")

df["Department"]=df["Department"].replace({"hr":"HR",
                                           "SALES":"Sales"})

df["SalaryBand"]=df["SalaryBand"].replace({"low":"Low",
                                           "HIGH":"High",
                                           "MED":"Medium"})


df["SalaryBand"]=df["SalaryBand"].fillna("other")

df["JobRole"].value_counts()

df["JobRole"]=df["JobRole"].replace({"MANAGER":"Manager",
                                     "MED":"Medium"}
)

df["JobRole"]=df["JobRole"].fillna("other")

df["Gender"] = df["Gender"].str.strip()

df["Gender"] = df["Gender"].replace({
    "male": "Male",
    "M": "Male",
    "F": "Female",
    "FEMALE": "Female",
    "other" : "Other"
})

df["Gender"]=df["Gender"].fillna("other")

df["YearsAtCompany"] = df["YearsAtCompany"].replace('Unknown', np.nan)
df["YearsAtCompany"] = df["YearsAtCompany"].astype(float)
df["YearsAtCompany"]=df["YearsAtCompany"].fillna(df["YearsAtCompany"].mean())

df["PerformanceScore"]=df["PerformanceScore"].fillna(df["PerformanceScore"].mean())

df["MonthlyIncome"]=df["MonthlyIncome"].replace("#ERROR",np.nan)

df["MonthlyIncome"]=df["MonthlyIncome"].replace("-1000",1000)

df["Attrition"]=df["Attrition"].replace({"1":"Yes",
                                         "y":"Yes",
                                         "YES":"Yes",
                                         "no":"No",
                                         "0":"No",
                                         np.nan:"Yes"
                                         })

df["Attrition"].value_counts()

print("\n==  CATEGORICAL VARIABLES  ==\n")
categorical_cols = ["Department","SalaryBand", "JobRole", "Gender", "Attrition"]

for col in categorical_cols:
    print(f"\n-- {col} --")
    print(df[col].value_counts())


    #VISUALIZATION

    plt.figure(figsize=(10,6))
    df[col].value_counts().plot(kind="bar",color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\n==  NUMERICAL VARIABLES  ==\n")

numerical_cols = [
    "Age","YearsAtCompany","PerformanceScore","MonthlyIncome"
    
]

for col in numerical_cols:

    df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\n--- {col} ---")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Std Dev: {df[col].std():.2f}")
    print(f"Min: {df[col].min():.2f}")
    print(f"Max: {df[col].max():.2f}")

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    df[col].hist(bins=30)
    plt.title(f"Distribution of {col}")

    plt.subplot(1,2,2)
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

department_performance = df.groupby('Department').agg({
    'MonthlyIncome': 'mean',
    'Age': 'mean',
    'EmployeeID': 'count'
}).rename(columns={
    'EmployeeID': 'Employee Count'
})

print(department_performance)

plt.figure(figsize=(12,8))

# Average Monthly Income
plt.subplot(2,2,1)
department_performance['MonthlyIncome'].plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title('Income Distribution by Department')

# Employee Count
plt.subplot(2,2,2)
department_performance['Employee Count'].plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title('Employee Distribution by Department')

# Employee Count Bar Chart
plt.subplot(2,2,3)
sns.barplot(
    x=department_performance.index,
    y=department_performance['Employee Count']
)
plt.title('Employee Count by Department')
plt.xticks(rotation=45)

# Average Age
plt.subplot(2,2,4)
sns.barplot(
    x=department_performance.index,
    y=department_performance['Age']
)
plt.title('Average Age by Department')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

jobrole_analysis = df.groupby("JobRole").agg({
    "MonthlyIncome": "mean",
    "Age": "mean",
    "PerformanceScore": "mean",
    "EmployeeID": "count"
}).rename(columns={"EmployeeID": "Employee Count"})


plt.figure(figsize=(12, 8))

# Employee Count
plt.subplot(2, 2, 1)
sns.barplot(
    x=jobrole_analysis.index,
    y=jobrole_analysis['Employee Count']
)
plt.title('Employee Count by Job Role')

# Average Monthly Income
plt.subplot(2, 2, 2)
sns.barplot(
    x=jobrole_analysis.index,
    y=jobrole_analysis['MonthlyIncome']
)
plt.title('Average Monthly Income by Job Role')

# Average Age
plt.subplot(2, 2, 3)
sns.barplot(
    x=jobrole_analysis.index,
    y=jobrole_analysis['Age']
)
plt.title('Average Age by Job Role')

# Average Performance Score
plt.subplot(2, 2, 4)
sns.barplot(
    x=jobrole_analysis.index,
    y=jobrole_analysis['PerformanceScore']
)
plt.title('Average Performance Score by Job Role')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

gender_analysis = df.groupby("Gender").agg({
    "MonthlyIncome": "mean",
    "Age": "mean",
    "PerformanceScore": "mean",
    "EmployeeID": "count"
}).rename(columns={"EmployeeID": "Employee Count"})


plt.figure(figsize=(12, 8))

# Employee Count
plt.subplot(2, 2, 1)
sns.barplot(
    x=gender_analysis.index,
    y=gender_analysis['Employee Count']
)
plt.title('Employee Count by Gender')

# Average Monthly Income
plt.subplot(2, 2, 2)
sns.barplot(
    x=gender_analysis.index,
    y=gender_analysis['MonthlyIncome']
)
plt.title('Average Monthly Income by Gender')

# Average Age
plt.subplot(2, 2, 3)
sns.barplot(
    x=gender_analysis.index,
    y=gender_analysis['Age']
)
plt.title('Average Age by Gender')

# Average Performance Score
plt.subplot(2, 2, 4)
sns.barplot(
    x=gender_analysis.index,
    y=gender_analysis['PerformanceScore']
)
plt.title('Average Performance Score by Gender')

plt.tight_layout()
plt.show()





