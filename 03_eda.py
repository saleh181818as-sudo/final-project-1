import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv("titanic_phase2.csv")  # load the processed dataset from Phase 2

b="Fare"  # choose Fare as the target column

c=["Age","Fare","SibSp"]
for d in c:
 plt.figure()
 a[d].hist(bins=30)
 plt.title(d)
 plt.xlabel(d)
 plt.ylabel("Count")
 plt.show()
# plot histograms for 3 numerical columns to see their distributions

e=["Pclass","Survived"]
for f in e:
 a.boxplot(column=b,by=f)
 plt.title(b+" by "+f)
 plt.suptitle("")
 plt.xlabel(f)
 plt.ylabel(b)
 plt.show()
# compare the target (Fare) across two categories using boxplots

g=a.select_dtypes(include=np.number).corr()  # compute correlation matrix for numeric columns
h=g[b].abs().sort_values(ascending=False).head(10).index  # get top 10 features most correlated with Fare
i=a[h].corr()  # create smaller correlation matrix for these top features

plt.figure()
sns.heatmap(i,cmap="coolwarm")
plt.title("Top 10 Correlation Heatmap")
plt.show()
# show heatmap for the top 10 features related to the target

plt.figure()
plt.scatter(a["Age"],a["Fare"],c=a["Survived"],s=a["Pclass"]*20)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
# scatter plot to study relationship between Age and Fare
# color depends on Survived and size depends on Pclass

j=a.groupby("Pclass")[b].mean()
print(j)
print(j.idxmax(),j.idxmin())
# compute the mean Fare for each passenger class
# then print which class has the highest and lowest mean Fare

k=a[b].to_numpy()
l=np.sum(k)/len(k)
m=np.sqrt(np.sum((k-l)**2)/len(k))
print(l)
print(m)
# manually calculate the mean and standard deviation of Fare using NumPy

n=a["Age"].to_numpy()
o=(n-np.mean(n))/np.std(n)
p=a["k"].to_numpy()
print(np.allclose(o,p))
# standardize Age manually using z-score formula
# compare it with the scaled Age column from Phase 2

q=a.loc[a[b].idxmax(),["Age","Fare","SibSp"]].to_numpy()
r=a.loc[a[b].idxmin(),["Age","Fare","SibSp"]].to_numpy()
s=np.dot(q,r)/(np.linalg.norm(q)*np.linalg.norm(r))
print(s)
# calculate cosine similarity between the highest Fare record and the lowest Fare record

t=a[b].median()
u=a[a["Pclass"]==1]
v=np.mean(u[b]>t)
print(v)
# estimate probability:
# among first-class passengers, what fraction have Fare above the median Fare

