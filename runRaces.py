import pandas as pd
import numpy as np



def getOdds(df, all_odds):
    df_odds = all_odds[all_odds['runner_id'].isin(df.index)]

    #I take the mean odds for the horse rather than the odds 1 hour before or 10 mins before. You may want to revisit this.
    average_win_odds = df_odds.groupby(['runner_id'])['odds_one_win'].mean()

    #delete when odds are 0 because there is no market for this horse
    average_win_odds[average_win_odds == 0] = np.nan
    df['odds'] = average_win_odds
    df = df.dropna(subset=['odds'])
    #given that I predict multiple winners, there's leakage if I don't shuffle the test set (winning horse   appears first and I put money on the first horse I predict to win)
    df = df.iloc[np.random.permutation(len(df))]
    return df


def train_test_split(df):
    training_races = np.random.choice(df['market_id'].unique(),size=int(round(0.7*len(df['market_id'].unique()),0)),replace=False)
    df_train = df[df['market_id'].isin(training_races)]
    df_test = df[~df['market_id'].isin(training_races)]
    return (df_train, df_test)

def fitModel(df_train, df_test, est):
    est.fit(df_train.drop(['win','position','market_id'],axis=1), df_train['win'])
    #if est.predict_proba:
    #to_predict = df_test.drop(['win','position','market_id'], axis=1)
    print(df_test.isnull().sum())
    predictions = est.predict_proba(df_test.drop(['win','position','market_id'],axis=1))[:,1]
    #else:
    #predictions = est.predict(df_test.drop(df_test[['win','position','market_id']],axis=1))[:,0]
    
    df_test['winProb'] = predictions
    df_test = df_test[['winProb','win','market_id']]
    return df_test

def placeBets(df):
    investment = 0
    payout = 0
    bets = 0
    wins = 0
    for index, row in df.iterrows():
        # if we predicted the probability of a win that's greater than the odds, we bet
        if row['winProb'] > (1/row['odds']):
            investment +=1
            bets +=1
            if (row['win']):
                payout += row['odds']
                wins +=1
                
    investment_return = round((payout - investment)/investment*100,2)
    print('Total bets: {}'.format(bets))
    print('Total wins: {}'.format(wins))
    return investment_return


def runRaces(data, est, allOdds, n_trials=1):
    data = data.dropna(axis=0, subset=['market_id']) # can't have market_ids with na.
    payouts = np.array([])
    for i in range(n_trials):
        print('trial {}'.format(i))
        df_train, df_test = train_test_split(data)
        df_test = fitModel(df_train, df_test, est)
        df_test = getOdds(df_test, allOdds)
        payout = placeBets(df_test)
        payouts = np.append(payouts, payout)
        
    print('mean payout: {}, std payout: {}'.format(np.mean(payouts), np.std(payouts)))
    return df_test 