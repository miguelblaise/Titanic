import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def cleandata(dataframe):
    S = []
    C = []
    Q = []
    upper = []
    middle = []
    lower = []
    sex = []
    hascabin = []
    
    for index, row in dataframe.iterrows():
        if row['Cabin'] == row['Cabin']:
            hascabin.append(1)
        else:
            hascabin.append(0)
        if row['Sex'] == 'male':
            sex.append(1)
        elif row['Sex'] == 'female':
            sex.append(0)
        else:
            sex.append(None)
        if row['Pclass'] == 1:
            upper.append(1)
            middle.append(0)
            lower.append(0)
        elif row['Pclass'] == 2:
            upper.append(0)
            middle.append(1)
            lower.append(0)
        elif row['Pclass'] == 3:
            upper.append(0)
            middle.append(0)
            lower.append(1)
        else:
            upper.append(0)
            middle.append(0)
            lower.append(0)
        if row['Embarked'] == 'S':
            S.append(1)
            C.append(0)
            Q.append(0)
        elif row['Embarked'] == 'C':
            S.append(0)
            C.append(1)
            Q.append(0)
        elif row['Embarked'] == 'Q':
            S.append(0)
            C.append(0)
            Q.append(1)
        else:
            S.append(0)
            C.append(0)
            Q.append(0)
    
    dataframe['Age'].fillna(dataframe['Age'].mean(), inplace = True)
    dataframe['Fare'].fillna(dataframe['Fare'].mean(), inplace = True)
    
    dataframe['Southampton'] = np.asarray(S)
    dataframe['Cherbourg'] = np.asarray(C)
    dataframe['Queenstown'] = np.asarray(Q)
    dataframe['Upper'] = np.asarray(upper)
    dataframe['Middle'] = np.asarray(middle)
    dataframe['Lower'] = np.asarray(lower)
    
    dataframe['SexB'] = np.asarray(sex)
    
    dataframe['HasCabin'] = np.asarray(hascabin)
    
    del dataframe['Embarked']
    del dataframe['Pclass']
    del dataframe['Sex']
    del dataframe['PassengerId']
    del dataframe['Name']
    del dataframe['Ticket']
    del dataframe['Cabin']
    
    return dataframe

traindata = cleandata(train)
test = cleandata(test)
iris = load_iris()
logClassifier = linear_model.LogisticRegression(C=1,   random_state=111)
logClassifier.fit(traindata[traindata.columns[1:]], traindata[traindata.columns[0]])
predicted = logClassifier.predict(test)
'''y_test = test[test.columns[0]]
y_test = y_test.values
accuracy = metrics.accuracy_score(y_test, predicted)
print(accuracy)'''
final = pd.read_csv("test.csv")
final = pd.DataFrame(final['PassengerId'])
predicted = predicted.tolist()
final.columns = ['PassengerId']
print(type(final))
final['Survived'] = np.asarray(predicted)
final.to_csv('Final.csv')
