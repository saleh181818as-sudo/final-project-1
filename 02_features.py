import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

a=pd.read_csv("cleaned_titanic.csv")  # load cleaned data from Phase 1

b=pd.get_dummies(a,columns=["Sex","Embarked"],drop_first=True)
# convert categorical columns (Sex, Embarked) into numbers using one-hot encoding

c={3:0,2:1,1:2}
b["d"]=b["Pclass"].map(c)
# convert Pclass into ordered numbers (1st class highest, 3rd lowest)

b["e"]=b["SibSp"]+b["Parch"]+1
# create family size feature (including the passenger)

b["f"]=np.where(b["e"]>0,b["Fare"]/b["e"],0)
# create ratio: Fare per person in family (safe division)

b["g"]=b["Age"]*b["d"]
# create interaction feature (Age × class)

plt.hist(b["Fare"],bins=30)
plt.show()
# show distribution of Fare before transformation

b["h"]=np.log1p(b["Fare"])
# apply log transformation to reduce skewness in Fare

plt.hist(b["h"],bins=30)
plt.show()
# show distribution after transformation

b["i"]=pd.cut(
    b["Age"],
    bins=[0,12,19,35,60,100],
    labels=["Child","Teen","Young","Adult","Old"],
    include_lowest=True
)
# group Age into categories (binning)

j=StandardScaler()
b[["k","l"]]=j.fit_transform(b[["Age","Fare"]])
# scale Age and Fare (standardization)

b=pd.get_dummies(b,columns=["i"],drop_first=True)
# convert age groups into numeric columns

m=b.select_dtypes(include=np.number).corr().abs()
# compute correlation between numeric features

n=m.where(np.triu(np.ones(m.shape),k=1).astype(bool))    #clean the matrix
# keep only upper triangle to avoid duplicate pairs and remove duplicate comparison

o=[p for p in n.columns if any(n[p]>0.95)]    # searches inside cleaned matrix to search columns have correlation more that 0.95 and if therer r column so simmilar add it into o
# find columns that r too similar to remove

b=b.drop(columns=o)
# remove columns that is in o

print(b.head())  # preview final data
print(b.shape)   # show size of dataset
print(b.info())  # show final structure

b.to_csv("titanic_phase2.csv",index=False)
# save processed data for Phase 3
