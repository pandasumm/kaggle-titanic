from dataCleaning import *
import pandas as pd

def result2csv(test, res, csvName):
    df_output = pd.DataFrame()
    df_output['PassengerId'] = test['PassengerId']
    df_output['Survived'] = res
    df_output[['PassengerId','Survived']].to_csv('../predictions/'+csvName+'.csv', index=False)

if __name__ == "__main__":
    dataset = pd.read_csv("../data/train.csv")
    # dataset = pd.read_csv('../data/test.csv')
    dataX, resY = dataCleaning(dataset)
    Validation(dataX, resY)
