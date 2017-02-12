import pandas as pd
import numpy as np

def createBaseFeatures():
    df_market = pd.read_csv("input/markets.csv")
    df_runners = pd.read_csv("input/runners.csv",dtype={'barrier': np.int16,'handicap_weight': np.float16})

    ##merge the runners and markets data frames
    df_runners_and_market = pd.merge(df_runners,df_market,left_on='market_id',right_on='id',how='outer')
    #df_runners_and_market.index = df_runners_and_market['id_x'] 
    df_runners_and_market = df_runners_and_market.rename(columns={'id_x': 'runner_id'})

    numeric_features = ['position', 'horse_id', 'runner_id', 'market_id','barrier','handicap_weight']

    df_features = df_runners_and_market[numeric_features]
    #turn the target variable into a binary feature: did or did not win
    df_features['win'] = False
    df_features.loc[df_features['position'] == 1,'win'] = True
    df_features.to_csv('input/base-features.csv', header=True, index=False)

def getBaseFeatures():
    features = pd.read_csv('input/base-features.csv')
    #randomly shuffle horses to make sure 
    #the winning horse for each race isn't always the first one
    return features.reindex(np.random.permutation(features.index))

if __name__ == "__main__":
    createBaseFeatures()