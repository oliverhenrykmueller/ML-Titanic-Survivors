import numpy as np
import sklearn
import csv
import pandas as pd
from sklearn import svm
import logging as lg
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


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

def data_explore(data):


    #data_male_dead = data[data["Sex"] == "male" & data["Survived"] == 0]
    data_filtered_age = data.loc[:,["Age","Survived","Sex"]]

    data_male = data[data["Sex"] == "male"]
    data_male_filt = data_male.loc[:,["Age","Survived","Fare","Pclass"]]
    print(data_male_filt)
    #data_male = data[data["Sex" == "male"]].loc[:,["Age","Survived","Fare","Pclass"]]

    data_male_age = data_filtered_age[data["Sex"] =="male"]
    data_female_age = data_filtered_age[data["Sex"] == "female"]

    data_male_dead = data[(data.Sex == "male") & data.Survived == 0]
    data_male_survived = data[(data.Sex == "male") & (data.Survived == 1)]
    data_female_dead = data[(data["Sex"] == "female")& (data["Survived"] == 0)]
#    data_female = data[(data["Sex"] == "female") & (data["S"])]

    print(data_male_dead)

    #sns.kdeplot(x="Age", y="Survived", data=data_male_age)

    # Comparing all male passengers Ages with 2 plots depending if they survived or not
    #sns.pairplot(data_male_age, hue="Survived",kind="hist")
    # Same as before but with female passengers
    #sns.pairplot(data_female_age, hue="Survived",kind="hist")

    sns.pairplot(data_male_filt,hue="Survived",kind="scatter")

    #sns.pairplot()

    #sns.histplot(data_male_age, hue="Survived")
    #sns.histplot(data_male_survived["Age"])



    #sns.distplot(data["Age"])


    plt.show()











    return

if __name__ == '__main__':

    ## OPEN TITANIC DATASET
    train = pd.read_csv("/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/train.csv")
    test = pd.read_csv("/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/test.csv")

    ## SHOW DATA FRAME
    print ("TRAINING DATA")
    print (train.head())
    print (train.describe())
    print (train.info())

    print ("TEST DATA")
    print (test.head())
    print (test.describe())
    print (test.info())

    ## Exploratory Data Analysis

    print (checkEmptyValues(train, "Survived"))
    print (checkEmptyValues(train, "Age"))
    print (checkEmptyValues(train, "Sex"))
    print (checkEmptyValues(train, "Fare"))
    print (checkEmptyValues(train, "SibSp"))
    print (checkEmptyValues(train, "Parch"))
    print (checkEmptyValues(train, "Pclass"))


    ## PREPROCESSING


    ## Drop unwanted columns
    # Save PassengerIDlist of Testset
    test_np_id = test["PassengerId"].to_numpy()
    print (test_np_id)

    train.drop(["PassengerId","Cabin", "Ticket", "Name"], axis=1, inplace=True)
    test.drop(["PassengerId","Cabin", "Ticket", "Name"], axis=1, inplace=True)


    # Drop rows with missing data


    # Imputation of missing values

    # Get the most repeated value of Embarked and replace missing rows with it
    embarked = train['Embarked'].mode()[0]

    data_complete = [train, test]
    for dataset in data_complete:
        dataset['Embarked'] = dataset['Embarked'].fillna(embarked)
    print (train.info())
    print (test.info())

    # Fare and Age values are missing for Testset
    data_merged = pd.concat([train.drop(["Survived"], axis=1), test], axis=0)

    # Mean and standard deviation for age
    mean_age = data_merged["Age"].mean()
    std_age = data_merged["Age"].std()

    # Age values
    for dataset in data_complete:

        Age_null_count = dataset["Age"].isnull().sum()

        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean_age - std_age, mean_age + std_age, size=Age_null_count)

        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age

        dataset["Age"] = age_slice
        dataset["Age"] = train["Age"].astype(int)

    print (train.info())
    print (test.info())

    # Fare Value

    # Get the mean for all Classes (even though only one is missing)
    fare_group = data_merged.groupby("Pclass").mean()
    pclass1_fare_mean = fare_group["Fare"][1]
    pclass2_fare_mean = fare_group["Fare"][2]
    pclass3_fare_mean = fare_group["Fare"][3]

    print (test)
    print (test[test['Fare'].isna()])

    # Fill in mean generated Class for missing values
    for x in test["Pclass"]:

        if x == 1:
            test["Fare"].fillna(pclass1_fare_mean,inplace=True)
        if x == 2:
            test["Fare"].fillna(pclass2_fare_mean, inplace=True)
        else:
            test["Fare"].fillna(pclass3_fare_mean, inplace=True)

    print (test.info())
    print (test)
    print (test[test['Fare'].isna()])

    ###################################################

    # Feature Engineering, create more significant features
    # Transformation of Categorical Data (Pclass = feature1(class1), feature2(Class2) etc.)

    data_explore(train)

    """
    # Feature Vector Processing
    # Scaling of numerical data, Caterogical data stays

    train_numerical_features = list(train.select_dtypes(include=['int64', 'float64', 'int32']).columns)
    del train_numerical_features[0]

    ss_scaler = StandardScaler()
    train_ss = pd.DataFrame(data=train)
    test_ss = pd.DataFrame(data=test)
    train_ss[train_numerical_features] = ss_scaler.fit_transform(train_ss[train_numerical_features])
    test_ss[train_numerical_features] = ss_scaler.transform(test_ss[train_numerical_features])

    print test_ss

    # One hot encoding, Dummy variables for categorical data to use in regression
    encode_col_list = list(train.select_dtypes(include=['object']).columns)
    for col in encode_col_list:
        train_ss = pd.concat([train_ss, pd.get_dummies(train_ss[col], prefix=col)], axis=1)
        test_ss = pd.concat([test_ss,pd.get_dummies(test_ss[col],prefix=col)],axis=1)
        train_ss.drop(col, axis=1, inplace=True)
        test_ss.drop(col,axis=1,inplace=True)

    # Turn String features into Numeric Values
    # train['Embarked'] = pd.factorize(train.Embarked)[0]
    # train["Sex"] = pd.factorize(train.Sex)[0]

    print train_ss
    print test_ss


    # Transform Data in Numpy format
    # Scikit learn datasets,train(X) and labels(y)
    X = train_ss.drop(["Survived"], axis=1).to_numpy()
    y = train["Survived"].to_numpy()
    X_test = test_ss.to_numpy()
    # kX = train.loc()
    print "FINAL TRAINING DATASET"
    print X
    print y
    print X_test

    ## Predictor Modeling ##

    # Setting up hyperparameters with Gridsearch

    parameters = {"penalty":["l1","l2"],
                  "C":[1.0,2.0,10.0,100.0],
                  "max_iter":[50,100,200,500]
                  }


    # Fitting the train data
    #clf = RandomForestClassifier(n_estimators=100)
    #clf.fit(X, y)

    logreg = LogisticRegression()
    scoring = ["r2","explained_variance"]
    grid = GridSearchCV(logreg, parameters, scoring=scoring, refit="r2", cv=5, iid=False)

    grid.fit(X,y)
    print grid
    #clf.fit(X, y)

    ## Prediction of test set
    #predictions = clf.predict(X_test)
    predictions = grid.predict(X_test)

    if len(test_np_id) != len(predictions):
        print "TEST SET AND PREDICTIONS NOT THE SAME LENGTHS"#
        print "POSSIBLE DELETED ROWS, ABORTING"
        raise Exception

    # Dstack for merging 2 1 dimensional arrays into one 2-dimensional one
    csv_predictions = np.dstack((test_np_id, predictions))[0]

    # Save the Numpyarray as a csv file
    with open("predictions.csv","w") as predictfile:
        cwriter = csv.writer(predictfile)
        cwriter.writerow(("PassengerId","Survived"))
    csv_file = open("predictions.csv", 'ab')
    np.savetxt(csv_file, csv_predictions, delimiter=',', fmt='%d')
    csv_file.close()
    """