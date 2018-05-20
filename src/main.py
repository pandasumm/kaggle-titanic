from dataCleaning import *
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def result2csv(test, res, csvName):
    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = res
    df_output[['PassengerId','Survived']].to_csv('../predictions/'+csvName+'.csv', index=False)

if __name__ == "__main__":
    dataset = pd.read_csv("../data/train.csv")
    A = pd.read_csv('../data/test.csv')
    resY = dataset.Survived

    dataX = dataCleaning(dataset.drop(['Survived'], axis=1))
    dataA = dataCleaning(A)

    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    # classifier = LogisticRegression(penalty='l2',random_state = 0)
    classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)

    classifier.fit(dataX, resY)
    accuracies = cross_val_score(estimator = classifier, X = dataX, y= resY, cv = 10)
    print("accuracies: ", accuracies.mean(), "+/-", accuracies.std(),"\n")

    # print(dataA)
    res = classifier.predict(dataA).astype(int)
    result2csv(A, res, "KNeighbors_result")
