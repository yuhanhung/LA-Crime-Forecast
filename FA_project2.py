#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:22:23 2018

@author: hungyuhan
"""
# Import the libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn import datasets

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#load the data - 187,029 cases
crime_all = pd.read_csv("/Users/hungyuhan/Desktop/2018 Spring/Fundamentals of Analytics/Final Project/Data/crime_python_v13.csv", low_memory=False)
crime_plot = pd.read_csv("/Users/hungyuhan/Desktop/2018 Spring/Fundamentals of Analytics/Final Project/Data/crime_python_v6_plot.csv", low_memory=False)
#show the header
crime_all.head()
#show the first two rows of contents
crime_all.iloc[:2]

# Importing the dataset
sns.set(font_scale=.6)
corr = crime_all.corr()
cover= np.zeros_like(corr)
cover[np.triu_indices_from(cover)]=True
# Heatmap
with sns.axes_style("white"):
    sns.heatmap(corr, mask=cover, square=True,cmap="RdBu", annot=True)

'''
#Descriptive statistics
fig, ax = plt.subplots()
plt.hist(crime_plot["Crime type"],orientation='horizontal')
plt.title("Crime Type Histogram")
plt.xlabel("Frequency")
plt.ylabel("Value")
fig.tight_layout()
plt.show()
'''

###Data clean
#drop the unneccessary attributes
#df_1=crime_all.drop(['Date Reported','Address','Latitude','Longitude','Description of crime category','Premise_Description','Weapon Use'],axis=1)

#drop all the rows with NAN (drop missing values) - 158,824 cases
#df_2=crime_all.dropna()

#delete the sex not female or male
df_3=crime_all[crime_all["Victim sex"] != "X"]
df_3=df_3[df_3["Victim sex"] != "H"]
df_3=df_3[df_3["Victim sex"] != "na"]
df_3=df_3[df_3["Victim age"] != -1]
df_3.iloc[:5]

#check types of variables
df_3.dtypes

#Replace the string variable
#df_3["Crime type"].value_counts()
df_3.groupby(['Month', 'Category of crime']).size()

#encode categorical variables
cat_col=["Category of crime","Month","Weekdays","Day of Month","Holiday","Time Occured","Area Code","zip code","Victim sex","Premise Code","Rain","Weapon Use"]
#cat_col=["Category of crime","Month","Day of Month","Time Occured","Premise Code","Weapon Use"]

for i in df_3.columns:
    if i in cat_col:
        df_3[i]=df_3[i].astype('category').cat.codes
df_3.head()


#change numeric variables into standardized number
#only can execute one attribute for one time, so copy the same line to repeat the normalization process on different attributes
#df_3["Income"]=StandardScaler().fit_transform(df_3["Income"].reshape(-1,1))
df_3["Victim age"]=StandardScaler().fit_transform(df_3["Victim age"].reshape(-1,1))

#set x & y
df_3.columns.values
X=df_3.columns.values.tolist() #generate the list of all attributes (only column name not data)
X.remove("Category of crime") #list of attributes 
Y="Category of crime"

#generate training and test dataset (adjust the test_size)
train_X,test_X,train_y,test_y=train_test_split(df_3[X],df_3[Y],test_size=0.2,random_state=0) #don't need to change random_state


#Support Vector Machine Classifier
model1=LinearSVC()
#Train the supervised model on the training set using .fit(X_train,y_train)
model1.fit(train_X,train_y)
y_pred1 = model1.predict(test_X)
print(metrics.classification_report(test_y,y_pred1))
print("Accuracy of SVM is ",accuracy_score(test_y, y_pred1))

#Random Forest
model2=RandomForestClassifier(n_estimators=100,n_jobs= -1)
#Train the supervised model on the training set using .fit(X_train,y_train)
model2=model2.fit(train_X, train_y)
y_pred2=model2.predict(test_X)
print(metrics.classification_report(test_y,y_pred2))
print("Accuracy of Random Forest is ",accuracy_score(test_y, y_pred2))
# model2.score(test_X, test_y) #->>> this line is the same as accuracy_score,accuracy: 0.8087122786599052

cm = confusion_matrix(test_y,model2.predict(test_X))
sns.heatmap(cm,annot=True,fmt="d")

feature_importances = pd.DataFrame(model2.feature_importances_,index = train_X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
'''
importance = model2.feature_importances_
std = np.std([tree.feature_importances_ for tree in model2.estimators_],
             axis=0)
indices = np.argsort(importance)[::-1]

importance = pd.DataFrame(importance, index=train_X.columns, 
                          columns=["Importance"])
imp_plot=importance.sort_values('Importance',ascending=False)
sns.factorplot(x="Importance", y="Index", data = imp_plot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')
'''
#importance["Std"] = np.std([tree.feature_importances_
#                            for tree in model2.estimators_], axis=0)
'''
x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

plt.bar(x, y, yerr=yerr, align="center")

plt.show()
'''
#Naive Bayes
clf = GaussianNB()
clf.fit(train_X, train_y)
target_pred = clf.predict(test_X)
print(metrics.classification_report(test_y,target_pred))
print("Accuracy of Naive Bayes is ",accuracy_score(test_y, target_pred))

#Decision Tree
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(train_X, train_y)
y_pred3 = clf_gini.predict(test_X)
print(metrics.classification_report(test_y,y_pred3))
print("Accuracy of Decision Tree is ", accuracy_score(test_y,y_pred3)*100)