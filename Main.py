import csv
import numpy as np
import pandas as pd
from sklearn import svm
import logging as lg
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import GridSearchCV
import re
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(style="ticks", color_codes=True)


def data_explore(data):
    data.Survived.replace({1: "Alive", 0: "Dead"}, inplace=True)
    print(data)

    # data_male_dead = data[data["Sex"] == "male" & data["Survived"] == 0]
    data_fare = data.loc[:, ["Fare", "Survived"]][data["Fare"] < 200]
    data_filtered_age = data.loc[:, ["Age", "Survived", "Sex"]]

    data_male = data[data["Sex"] == "male"]
    data_female = data[data["Sex"] == "female"]

    data_male_age = data_filtered_age[data_filtered_age["Sex"] == "male"]
    data_female_age = data_filtered_age[data_filtered_age["Sex"] == "female"]

    data.groupby('Sex').mean()
    FacetGrid = sns.FacetGrid(data, row='Embarked', size=4.5, aspect=1.6)
    FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None)
    FacetGrid.add_legend();

    """
    data_male_filt = data_male.loc[:,["Age","Survived","Fare","Pclass"]]
    print(data_male_filt)
    #data_male = data[data["Sex" == "male"]].loc[:,["Age","Survived","Fare","Pclass"]]

    data_male_age = data_filtered_age[data["Sex"] =="male"]
    data_female_age = data_filtered_age[data["Sex"] == "female"]

    data_male_dead = data[(data.Sex == "male") & data.Survived == 0]
    data_male_survived = data[(data.Sex == "male") & (data.Survived == 1)]
    data_female_dead = data[(data["Sex"] == "female")& (data["Survived"] == 0)]
#    data_female = data[(data["Sex"] == "female") & (data["S"])]
    """
    # print(data_male_dead)

    # print(data_female_age)

    # sns.kdeplot(x="Age", y="Survived", data=data_male_age)

    # Comparing all male passengers Ages with 2 plots depending if they survived or not
    # sns.pairplot(data_male_age, hue="Survived",kind="hist")
    # sns.pairplot(data_male_age, hue="Survived", diag_kind="hist", diag_kws={'alpha': 0.5, 'bins': 30})

    # signifacant male death age groups:
    # 0-5 Highest survival ratio
    # good survival rate until 16, catastrophic rate there
    # relaxing at 24-29
    # getting better at 32-36
    # worsening ab 37
    # good ratio at 48-50
    # then bad
    # 64-74 terrible
    # high death rate until 77-80 which survived colmpletly

    # Same as before but with female passengers

    # relative bad surviverate for infants < 2
    # good rate 2-6
    # 6-8 bad (half)
    # most terrible death rate for 8-12
    # 12 -16 good survival
    # 16 -20 worsening
    # good consistent rates for 20-31
    # very good rates for 31-36(38)
    # worsening rates until 44
    # bad rate at 44-48
    # very good rates until 56
    # actually until 63
    # sns.jointplot(x="Age", y="Survived", data=data, hue="Sex", kind="hist")

    # sns.pairplot(data_female_age, hue="Survived",diag_kind="hist",diag_kws = {'alpha':0.55, 'bins':30})
    # sns.histplot(data_female,bins=25,hue="Survived")

    # sns.pairplot(data_male_filt,hue="Survived",kind="scatter")

    # sns.histplot(data_male_age, hue="Survived")
    # sns.histplot(data_male_survived["Age"])

    # sns.pairplot(data_male, hue="Survived",diag_kind="hist",diag_kws = {'alpha':0.55, 'bins':30})

    sns.pairplot(data_fare, hue="Survived", diag_kind="hist", diag_kws={'alpha': 0.55, 'bins': 30})

    # sns.jointplot(x="")

    plt.show()

    return


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)

    # Open Titanic Data  #################################################
    train = pd.read_csv(
        "/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/train.csv")
    test = pd.read_csv(
        "/media/Jung/Desktop/Arbeit & Karierre/Data Analytics & CS Projekte/Kaggle Competitions/Titanic/Data/test.csv")


    # Show information of the data #################################################
    print("TRAINING DATA")
    print(train.head())
    print(train.describe())
    print(train.info())

    print("TEST DATA")
    print(test.head())
    print(test.describe())
    print(test.info())

    # Preprocessing #################################################
    # Merge both test and train data for additional imputation data
    data_merged = pd.concat([train.drop(["Survived"], axis=1), test], axis=0)
    data_complete = [train, test, data_merged]

    # Save passengerID-list from testdata for later ###
    test_np_id = test["PassengerId"].to_numpy()



    # Imputation of missing values ###

    # "Embarked" column: Get the most repeated value of Embarked and replace missing rows with it #
    embarked = train['Embarked'].mode()[0]
    for dataset in data_complete:
        dataset['Embarked'] = dataset['Embarked'].fillna(embarked)
    print(train.info())
    print(test.info())

    # "Age" column: Get mean and standard deviation to compute random values in a range #
    # Mean and standard deviation for age
    mean_age = data_merged["Age"].mean()
    std_age = data_merged["Age"].std()

    for dataset in data_complete:
        Age_null_count = dataset["Age"].isnull().sum()

        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean_age - std_age, mean_age + std_age, size=Age_null_count)

        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age

        dataset["Age"] = age_slice
        dataset["Age"] = train["Age"].astype(int)

    print(train.info())
    print(test.info())

    # "Fare" column: Get the mean for all Classes (even though only one value is missing) #
    fare_group = data_merged.groupby("Pclass").mean()
    pclass1_fare_mean = fare_group["Fare"][1]
    pclass2_fare_mean = fare_group["Fare"][2]
    pclass3_fare_mean = fare_group["Fare"][3]

    # Fill in mean generated class for missing values
    # inefficient
    for dataset in data_complete:
        for x in dataset["Pclass"]:
            if x == 1:
                test["Fare"].fillna(pclass1_fare_mean, inplace=True)
            if x == 2:
                test["Fare"].fillna(pclass2_fare_mean, inplace=True)
            else:
                test["Fare"].fillna(pclass3_fare_mean, inplace=True)


    # Exploraty Data Analysis #################################################

    # data_explore(train)

    #raise Exception

    # Feature Engineering #################################################
    # Transformation of Categorical Data ##

    # New Feature: Familysize :
    # Create new Feature regarding family size, on the idea that families stick together
    # Combine Sibsp and Parch to count the children, parents etc. together
    for dataset in data_complete:
        dataset["Familysize"] = dataset["SibSp"] + dataset["Parch"]
        dataset.loc[dataset['Familysize'] > 0, 'travelled_alone'] = 'No'
        dataset.loc[dataset['Familysize'] == 0, 'travelled_alone'] = 'Yes'

    # New Feature: Age ranges
    for dataset in data_complete:
        ## OLD
        # dataset.loc[dataset["Age"] <= 2 , "Age_infant"] = True
        # dataset.loc[(dataset["Age"] > 2) & (dataset["Age"] <= 12), "Age_child"] = True
        # dataset.loc[(dataset["Age"] > 12 )& (dataset["Age"] <= 18), "Age_teen"] = True
        # dataset.loc[(dataset["Age"] > 18 )& (dataset["Age"] <= 30), "Age_youngadult"] = True
        # dataset.loc[(dataset["Age"] > 30 )& (dataset["Age"] <= 50), "Age_adult"] = True
        # dataset.loc[(dataset["Age"] > 50), "Age_oldadult"] = True

        # dataset["Age_infant"] = dataset["Age"] < 2
        # dataset["Age_child"] = (dataset["Age"] >= 2) & (dataset["Age"] < 13)
        # dataset["Age_teen"] = (dataset["Age"] >= 12) & (dataset["Age"] < 18)
        # dataset["Age_youngadult"] = (dataset["Age"] >= 18 ) & (dataset["Age"] < 30)
        # dataset["Age_adult"] = (dataset["Age"] >= 30 ) & (dataset["Age"] < 50)
        # dataset["Age_oldadult"] = dataset["Age"] >= 50

        # dataset.loc[dataset["Age"] < 2 ] = "Infant"

        # Categorize Age data first with number, since multiple edits are neccessary
        # and pandas cant compare strings with integers
        dataset["Age_num"] = dataset["Age"]

        dataset.loc[dataset["Age"] < 2, "Age"] = 0
        dataset.loc[(dataset["Age"] >= 2) & (dataset["Age"] < 6), "Age"] = 1
        dataset.loc[(dataset["Age"] >= 6) & (dataset["Age"] < 12), "Age"] = 2
        dataset.loc[(dataset["Age"] >= 12) & (dataset["Age"] < 16), "Age"] = 3
        dataset.loc[(dataset["Age"] >= 16) & (dataset["Age"] < 21), "Age"] = 4
        dataset.loc[(dataset["Age"] >= 21) & (dataset["Age"] < 30), "Age"] = 5
        dataset.loc[(dataset["Age"] >= 30) & (dataset["Age"] < 40), "Age"] = 6
        dataset.loc[(dataset["Age"] >= 40) & (dataset["Age"] < 48), "Age"] = 7
        dataset.loc[(dataset["Age"] >= 48) & (dataset["Age"] < 60), "Age"] = 8
        dataset.loc[(dataset["Age"] >= 60), "Age"] = 9
        dataset['Age'].replace(
            {0: 'Infant', 1: 'Small_Child', 2: "Child", 3: "Teen", 4: "Teen_Adult", 5: "Young_Adult", 6: "Adult",
             7: "Middleage_Adult", 8: "Old_adult", 9: "Senior"}, inplace=True)

    # New Feature: Deck Level, depending on the First letter, missing data will be filled with "U" with unknown

    a = train.sort_values("Cabin")["Cabin"]

    for dataset in data_complete:
        dataset['Decklevel'] = dataset['Cabin'].str.replace(r'\d', '')
        dataset.Decklevel.fillna("U", inplace=True)
        for row in dataset.Decklevel.items():
            level = row[1][0]
            index = row[0]
            dataset.at[index, "Decklevel"] = level

    # Drop the Cabin column since its not useful anymore
    for dataset in data_complete:
        dataset.drop("Cabin", axis=1)

    # New feature: Fare categorization #
    # Fare per Person since Fare counts for whole Families#
    # Look at data how Fare is handled in relatives
    print(train.loc[train["Familysize"] > 4])
    print(train.loc[train.Name.str.startswith("Asplund")])

    for dataset in data_complete:
        dataset["FarePerson"] = dataset["Fare"] / (dataset["Familysize"]+1)

    data_merged.hist("Fare", bins=50)
    #plt.show()

    # Binning 2 ranges of data, the low ones, and the medium to high ones
    lowfares = pd.qcut(data_merged.loc[data_merged["Fare"]< 90 ]["Fare"], 4)
    medium_highfares = pd.qcut(data_merged.loc[(data_merged["Fare"] > 90) & (data_merged["Fare"] < 250)]["Fare"], 2)
    print(lowfares.value_counts())
    print(medium_highfares.value_counts())

    # Data is mostly  distributed at 0-90, then lightly populated from ~90-250, finally has an outlier at 500

    for dataset in data_complete:
        dataset.loc[dataset["Fare"] < 7.896, "Fare"] = 0
        dataset.loc[(dataset["Fare"] >= 7.896) & (dataset["Fare"] < 13.0), "Fare"] = 1
        dataset.loc[(dataset["Fare"] >= 13.0) & (dataset["Fare"] < 26.55), "Fare"] = 2
        dataset.loc[(dataset["Fare"] >= 26.55) & (dataset["Fare"] < 89.104), "Fare"] = 3
        dataset.loc[(dataset["Fare"] >= 89.104) & (dataset["Fare"] < 146.521), "Fare"] = 4
        dataset.loc[(dataset["Fare"] >= 146.521) & (dataset["Fare"] < 247.521), "Fare"] = 5
        dataset.loc[(dataset["Fare"] >= 247.521), "Fare"] = 6
        dataset['Fare'].replace(
            {0: 'Extremely_low', 1: 'Very_Low', 2: "Low", 3: "Medium", 4: "High", 5: "Very High", 6: "Extremely_high"}, inplace=True)

    print(train.head())

    # New Feature: Title
    # Like Miss, Mr., Mrs and rarer titles like Sir, military ranks, etc.

    titles = train.Name.str.extract(r"([A-Za-z]+)\.")
    titles_higher = ["Dr", "Rev", "Major", "Col", "Mlle", "Mme", "Ms", "Capt", "Lady", "Jonkheer", "Don", "Countess", "Sir"]

    for dataset in data_complete:
        dataset["Title"] = dataset.Name.str.extract(r"([A-Za-z]+)\.", expand=False)
        dataset["Title"] = dataset.Title.replace(titles_higher, "High_title")

    print(train.Title.unique())


    # Feature Vector Processing #################################################

    ## Drop all uneccessary columns
    print("Feature Vector Processing")
    print(train)


    for dataset in data_complete:
        dataset.drop(["Cabin","Name","PassengerId", "Ticket"], axis=1,inplace=True)

    # Transformation of categorical data to prepare for One hot encoding ##
    # Pclass
    for dataset in data_complete:
        dataset.loc[dataset["Pclass"] == 1, "Pclass"] = "1"
        dataset.loc[dataset["Pclass"] == 2, "Pclass"] = "2"
        dataset.loc[dataset["Pclass"] == 3, "Pclass"] = "3"

    print(train)

    # Feature selection #########


    # Scaling numerical data ##

    numerical_features = list(train.drop(["Survived"], axis=1).select_dtypes(include=['int64', 'float64', 'int32']).columns)
    print(numerical_features)

    ss_scaler = StandardScaler()
    #ss_scaler = MinMaxScaler()
    train_ss = pd.DataFrame(data=train)
    test_ss = pd.DataFrame(data=test)

    ss_scaler.fit(train[numerical_features])

    train_ss[numerical_features] = ss_scaler.transform(train_ss[numerical_features])
    test_ss[numerical_features] = ss_scaler.transform(test_ss[numerical_features])

    print("DATA SCALED")

    print(train.select_dtypes(include=['object']).columns)

    # One hot encoding, Dummy variables for categorical data to use in regression ##
    encode_col_list = list(train.select_dtypes(include=['object']).columns)

    for col in encode_col_list:
        train_ss = pd.concat([train_ss, pd.get_dummies(train_ss[col], prefix=col)], axis=1)
        test_ss = pd.concat([test_ss, pd.get_dummies(test_ss[col], prefix=col)], axis=1)
        train_ss.drop(col, axis=1, inplace=True)
        test_ss.drop(col, axis=1, inplace=True)

    # OLD: Turn String features into Numeric Values
    # train['Embarked'] = pd.factorize(train.Embarked)[0]
    # train["Sex"] = pd.factorize(train.Sex)[0]

    print(train_ss)
    print(test_ss)

    # Transform Data in Numpy format ##
    # Scikit learn datasets,train(X) and labels(y)
    X = train_ss.drop(["Survived"], axis=1).to_numpy()
    y = train["Survived"].to_numpy()
    X_test = test_ss.to_numpy()
    print("FINAL TRAINING DATASET")
    print(X)
    print(y)
    print(X_test)

    ## Predictor Modeling #################################################

    # Setting up hyperparameters for Gridsearch

    parameters_reg = {"penalty": ["l2"],
                  "C": [1.0, 2.0, 10.0, 100.0],
                  "max_iter": [300,500, 1000, 2000]
                  }

    parameters_svm = {"loss" : ["hinge","squared_hinge"],
                      "C":[1.0,1.5,2.0,5,10],
                      "max_iter":[1000,2000,5000,10000,20000]
                      }

    parameters_svm = {"kernel": ["linear", "rbf","sigmoid"],
                      "C": [1.0, 1.5, 2.0, 5, 10],
                      "gamma":["scale","auto"]
                      }

    # Fitting the train data
    clf = RandomForestClassifier(n_estimators=100)

    clf_svc = svm.SVC()
    clf_linear_svm = svm.LinearSVC()

    grid_svm = GridSearchCV(clf_svc,param_grid=parameters_svm)
    #grid_svm = svm.LinearSVC()

    logreg = LogisticRegression()
    scoring = ["r2", "explained_variance"]
    #grid = GridSearchCV(logreg, parameters, scoring=scoring, refit="r2", cv=5)



    grid_svm.fit(X, y)

    # Prediction of test set ##
    #predictions = clf.predict(X_test)
    predictions = grid_svm.predict(X_test)

    # Check if initial testset(id) and predictions have the same length
    if len(test_np_id) != len(predictions):
        print("TEST SET AND PREDICTIONS NOT THE SAME LENGTHS")
        print("POSSIBLE DELETED ROWS, ABORTING")
        raise Exception

    # Dstack for merging 2 1 dimensional arrays into one 2-dimensional one
    # Merge predictions and the list with the PassengerIds
    csv_predictions = np.dstack((test_np_id, predictions))[0]

    # Save the Predictions as a csv file
    with open("predictions.csv", "w") as predictfile:
        cwriter = csv.writer(predictfile)
        cwriter.writerow(("PassengerId", "Survived"))
        np.savetxt(predictfile, csv_predictions, delimiter=',', fmt='%d')

    #csv_file = open("predictions.csv", 'ab')
    #csv_file.close()

    print(grid_svm.best_estimator_)
    print(grid_svm.best_params_)