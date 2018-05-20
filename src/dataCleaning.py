import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



def dataCleaning(dataset):
    dataX = dataset.drop(['PassengerId', 'Cabin', 'Ticket', 'Fare', 'Parch', 'SibSp'], axis=1)
    # vector of labels (dependent variable)
    # remove the dependent variable from the dataframe X
    # dataX = dataX.drop(['Survived'], axis=1)

    # labelEncoder_X = LabelEncoder()
    dataX.Sex = LabelEncoder().fit_transform(dataX.Sex)

    # encode "Embarked"
    # number of null values in embarked:
    # print ('Number of null values in Embarked:', sum(X.Embarked.isnull()))
    # fill the two values with one of the options (S, C or Q)

    row_index = dataX.Embarked.isnull()
    dataX.loc[row_index, 'Embarked'] = 'S'

    Embarked = pd.get_dummies(dataX.Embarked, prefix='Embarked')
    # drop one of the clowns
    # Embarked = pd.get_dummies(X.Embarked, prefix='Embarked', drop_first = True)
    dataX = dataX.drop(['Embarked'], axis=1)
    dataX = pd.concat([dataX, Embarked], axis=1)

    # -------- Change Name -> Title ----------------------------
    got = dataX.Name.str.split(',').str[1]
    dataX.iloc[:, 1] = pd.DataFrame(got).Name.str.split('\s+').str[1]

    #------------------ Average Age per title -------------------------------------------------------------
    ax = plt.subplot()
    ax.set_ylabel('Average age')
    dataX.groupby('Name').mean()['Age'].plot(kind='bar', figsize=(13, 8), ax=ax)

    title_mean_age = []
    # set for unique values of the title, and transform into list
    title_mean_age.append(list(set(dataX.Name)))
    title_mean_age.append(dataX.groupby('Name').Age.mean())
    # print("title_mean_age: ")
    # print(title_mean_age)
    #------------------------------------------------------------------------------------------------------

    #------------------ Fill the missing Ages ---------------------------
    n_traning = dataset.shape[0]  # number of rows
    print("rows: ", n_traning)
    n_titles = len(title_mean_age[1])
    for i in range(0, n_traning):
        # if np.isnan(dataX.Age[i]):
        if pd.isnull(dataX.at[i, 'Age']):
            for j in range(0, n_titles):
                if dataX.Name[i] == title_mean_age[0][j]:
                    # dataX.Age[i] = title_mean_age[1][j]
                    dataX.at[i, 'Age'] = title_mean_age[1][j]

    #--------------------------------------------------------------------
    dataX = dataX.drop(['Name'], axis=1)
    return dataX


def Validation(dataX, resY):

    #-----------------------Logistic Regression---------------------------------------------
    # # Fitting Logistic Regression to the Training set
    # from sklearn.linear_model import LogisticRegression
    # classifier = LogisticRegression(penalty='l2',random_state = 0)

    #-----------------------------------K-NN --------------------------------------------------

    # # Fitting K-NN to the Training set
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)

    #---------------------------------------SVM -------------------------------------------------

    # # Fitting Kernel SVM to the Training set
    # from sklearn.svm import SVC
    # classifier = SVC(kernel = 'rbf', random_state = 0)

    #---------------------------------Naive Bayes-------------------------------------------

    # Fitting Naive Bayes to the Training set
    # from sklearn.naive_bayes import GaussianNB
    # classifier = GaussianNB()

    #----------------------------Random Forest------------------------------------------

    # Fitting Random Forest Classification to the Training set

    classifier_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier_RF.fit(dataX, resY)
    # accuracies = cross_val_score(estimator = classifierRF, X=X , y=y , cv = 10)
    accuracies = cross_val_score(estimator=classifier_RF, X=dataX, y=resY, cv=10)
    print("Random Forest accuracies: ", accuracies.mean(), "+/-", accuracies.std(), "\n")
    # res = classifier.predict(A).astype(int)
