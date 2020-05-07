#!/usr/bin/env python
# coding: utf-8

# ## Loading and Analysis of data

# In[65]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[66]:


train_raw = pd.read_csv('train.csv',index_col='Loan_ID')
test_main = pd.read_csv('test.csv',index_col='Loan_ID')
train_raw.info()


# ## Separating the output column and processing data
# After separating output column both the data will be appended and preprocessing will be done

# In[67]:


#Taking the label encoded values in y 
y = train_raw.Loan_Status.map({'Y':1,'N':0})
train_raw.drop('Loan_Status',inplace=True,axis=1)


# In[68]:


rown_train_data = train_raw.shape[0]
#appending both the data
X = train_raw.append(test_raw)
X.head()


# In[69]:


objList = X.select_dtypes(include = "object").columns
for obj_Column in objList:
    print(X[obj_Column].unique())


# ## Label encoding the categorial faetures
# As the values also contain NAN so there will be selective label encoding followed by imputing

# In[72]:


# Encoders for the training data
encoders = dict()

for col_name in X.columns:
        series = X[col_name]
        label_encoder = LabelEncoder()
        X[col_name] = pd.Series(
            label_encoder.fit_transform(series[series.notnull()]),
            index=series[series.notnull()].index
        )
        encoders[col_name] = label_encoder

X.info()


# In[75]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

#Converting again to dataframe
X = pd.DataFrame(X)


# ## Splitting the data into train and validation sets 

# In[80]:


train_X = X.iloc[:rown_train_data,]
final_testing_data = X.iloc[rown_train_data:,]
seed=7
#getting columns back 
train_X.columns = test_raw.columns 
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(train_X,y,random_state=seed)


# # Robustly checking for which algorithm will perform best here

# In[107]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("svc",SVC()))
models.append(("rndf",RandomForestClassifier()))

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
name=[]


# In[ ]:



for name,model in models:
    cv_result=cross_val_score(model,train_X,train_y,cv=10,scoring='accuracy')
    result.append(cv_result.mean())
    names.append(name)

#printing all the results
for result,name in zip(name,result):
    print(name)
    print(result)


# ## Conclusion : Logistic Regression and Random Forest perform equally well
# Hence Ada boosting with the Random Forest Classifier : hyper parameter optimised 

# In[166]:


from sklearn.ensemble import AdaBoostClassifier
check = [20,30,40,45,50,55,60,70,100,150]
for estimators in check:
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=10,min_samples_leaf=28), n_estimators=estimators,
        algorithm="SAMME.R", learning_rate=0.25)
    ada_clf.fit(train_X,train_y)
    print("Score for estimator {} train data{}".format(estimators,ada_clf.score(train_X,train_y)))
    print("Test data {}".format(ada_clf.score(test_X,test_y)))


# ## Measuring the accuracy of the model

# In[168]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pred = ada_clf.predict(test_X)
print(accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))


# In[ ]:




