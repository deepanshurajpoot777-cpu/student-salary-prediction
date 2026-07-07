
# ==========================
# STEP 1 : Load Dataset
# ==========================

import pandas as pd
df=pd.read_csv("student_salary_Dataset.csv")

print("Sample Rows")
print(df.head())

print("\n DataSet Info")
print(df.info())

print("\nDATAset shape")
print(df.shape)

print("\nSummary Statics")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# STEP 2 : Data Cleaning
# ==========================


from sklearn.preprocessing import LabelEncoder
# Numerical Columns
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Years of Experience"]=df["Years of Experience"].fillna(df["Years of Experience"].mean())
df["Salary"]=df["Salary"].fillna(df["Salary"].mean())

# categorical columns

df["Gender"]=df["Gender"].fillna(df["Gender"].mode()[0])
df["Education Level"]=df["Education Level"].fillna(df["Education Level"].mode()[0])
df["Job Title"]=df["Job Title"].fillna(df["Job Title"].mode()[0])


print("\n mising Data set")
print(df.isnull().sum())

le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])  # yes =1 ,no=1
df=pd.get_dummies(df,columns=["Education Level"],drop_first=True,dtype=int)
df=pd.get_dummies(df,columns=["Job Title"],drop_first=True,dtype=int)

print("\n AFTER ENCODING")
print(df.tail())

print("\n DATA TYPE")
print(df.dtypes)

#===========================
# STEP 3 : Model Training
# ==========================


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns
print(df.columns)
# Feature and Target

x=df.drop("Salary",axis=1)
y=df["Salary"]

print("/n Features")
print(x.head())

print("Target")
print(y.head())

# train test split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Feature scalling

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# model Training

model=LinearRegression()
model.fit(x_train,y_train)

# prediction

y_pred=model.predict(x_test)
print("\n ACtual Price")
print(y_test.head())

print("\n Predict")
print(y_pred[:5])

# EValuation

print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score:",r2_score(y_test,y_pred))

# Visulization
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.title("Actual vs predicted Salary")
plt.ylabel("predicted salary")
plt.xlabel("Actual Salary")
plt.savefig("Actual_vs_predicted_salary.png")
plt.show()

# ==========================
# STEP 4 : User Prediction
# ==========================


print("\n predicted salary")

age=int(input("enter your Age:"))
gender=input("enter your Gender(Male/Female):")
education=input("enter your education Level")
job=input("enter your job Title:")
experience=int(input("enter your Year of experirnce:"))


user_input=pd.DataFrame(0,index=[0],columns=x.columns)

user_input["Age"]=age
user_input["Years of Experience"]=experience
# Gender encoding
user_input["Gender"]=1 if gender.lower()=="male" else 0

# education level encoding
edu_col="Education Level_"+education

if edu_col in user_input.columns:
    user_input[edu_col]=1

# job Title Encoding

job_col="Job Title_"+job

if job_col in user_input.columns:
    user_input[job_col]=1

# Feature scalling

user_scaled=scaler.transform(user_input)

# prediction

prediction=model.predict(user_scaled)

print("\n predicted salary :₹",round(prediction[0],2))