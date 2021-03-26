import numpy as np
import sklearn
import csv
import pandas as pd
from sklearn import svm
import logging as lg

def preprocess (data):

    return ""

def checkEmptyValues(df,column):

    array = []
    i = 0
    #print pd.isna(df["PassengerId"])

    for x in pd.isna(df[column]):
        if x == True:
            array.append(x)
    if not array:
        empty = True
    else:
        empty = False

    return empty,array

def genderbender(df):

    i = 0
    for item in df["Sex"]:
        if item == "male":
            df.at[i, "Sex"] = 0
        else:
            df.at[i, "Sex"] = 1
        i += 1

    return df

if __name__ == '__main__':

    ## OPEN TITANIC DATASET
    train = pd.read_csv("/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/train.csv")
    test = pd.read_csv("/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/test.csv")

    ## SHOW DATA FRAME
    print train.head()
    print train.describe()

    ## DROP COLUMNS
    train.drop(["Cabin","Ticket","Name","Embarked"],axis=1,inplace=True)

    ## DATA EXPLORATION

    #print train.groupby("Age").mean()

    print checkEmptyValues(train, "Survived")
    print checkEmptyValues(train, "Age")
    print checkEmptyValues(train, "Sex")
    print checkEmptyValues(train, "Fare")
    print checkEmptyValues(train, "SibSp")
    print checkEmptyValues(train, "Parch")
    print checkEmptyValues(train, "Pclass")


    ## PREPROCESSING

    #Change gender to number

    train = genderbender(train)
    print train



    #DELETE ROWS

    # AGE: DELETE ROW WTH MISSING VALUE

    train.dropna(subset=["Age"],inplace=True)

    #IMPUTATION


    # Scikit learn datasets,train(X) and labels(y)
    X = train.drop(["Sex","PassengerId","Survived"],axis=1).to_numpy()
    y = train["Survived"].to_numpy()
    X = train.loc()

    print X
    print y


    # Feature Vector Processing
    # Scaling



    ## Predictor Modeling ##
    # Fitting the train data
    clf = svm.SVC()
    clf.fit(X,y)

    print test.head()

    ## Preprocessing test data set
    test = genderbender(test)
    # Impute values in Age and Fare
    test.dropna(subset=["Age"],inplace=True)
    test.dropna(subset=["Fare"],inplace=True)


    # Drop unneccessary columns and CONVERT to NUMPY arrays for scikit-learn
    test_np_id = test["PassengerId"].to_numpy()
    test_np = test.drop(["Sex","PassengerId","Cabin","Ticket","Name","Embarked"],axis=1).to_numpy()

    print test_np

    ## Prediction of test set
    predictions = clf.predict(test_np)
    print test_np_id
    print predictions

    if len(test_np_id) != len(predictions):
        print "TEST SET AND PREDICTIONS NOT THE SAME LENGTHS"#
        print "POSSIBLE DELETED ROWS, ABORTING"
        raise Exception

    csv_predictions = np.dstack((test_np_id,predictions))
    print csv_predictions
    #for i in xrange(len(test_np_id))





