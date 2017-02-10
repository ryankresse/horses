import pandas as pd
import numpy as np



def addOdds(classAndProb, all_odds):
    df_odds = all_odds[all_odds['runner_id'].isin(classAndProb.runner_id)]
    average_win_odds = df_odds.groupby(['runner_id'])['odds_one_win'].mean()
    average_win_odds[average_win_odds == 0] = np.nan

    average_win_odds = pd.DataFrame(average_win_odds).reset_index().rename(columns={'odds_one_win':'odds'})
    average_win_odds = average_win_odds.dropna(subset=['odds'])

    return classAndProb.merge(average_win_odds, on='runner_id')



def trainTestSplit(raceData):
    market_ids = raceData[:,-1]
    training_races = np.random.choice(market_ids,size=int(round(0.7*market_ids.shape[0],0)),replace=False)
    trainMask = np.in1d(market_ids,training_races)
    train = raceData[trainMask]
    test = raceData[~trainMask]
    return (train, test)

def fitModel(train, test,  est=None, runnersClasses=None ):
    
    print('fitting estimator')
    est.fit(train[:, 0:-3], train[:, -2])
    
    print('making predictions')
    probs = est.predict_proba(test[:, 0:-3])

    colsToAdd = list(np.arange(probs.shape[1]))
    colsToAdd.append('market_id')
    market_ids = test[:, -1].T

    probs = np.hstack((probs, market_ids[:, np.newaxis]))
    probs_df = pd.DataFrame(probs, columns=colsToAdd)

    probs_re = probs_df.set_index('market_id').stack().reset_index().rename(columns={0:'prob', 'level_1':'class'})

    probs_re['class'] = probs_re['class'].astype(float)

    classAndProb = probs_re.merge(runnersClasses, on=['market_id', 'class'])    
    return classAndProb

def placeBets(df):
    investment = 0
    payout = 0
    bets = 0
    wins = 0
    for index, row in df.iterrows():
        # if we predicted the probability of a win that's greater than the odds, we bet
        if row['prob'] > (1/row['odds']):
            investment +=1
            bets +=1
            if (row['win']):
                payout += row['odds']
                wins +=1
                
    investment_return = round((payout - investment)/investment*100,2)
    print('Total bets: {}'.format(bets))
    print('Total wins: {}'.format(wins))
    return investment_return


def runRaces(data=None, runnersClasses=None, winOrNot=None, est=None, allOdds=None, n_trials=1):
    payouts = np.array([])
    for i in range(n_trials):
        print('trial {}'.format(i))
        train, test = trainTestSplit(data)
        classAndProb = fitModel(train, test, est=est, runnersClasses=runnersClasses)
        classAndProb = addOdds(classAndProb, allOdds)
        classAndProb = classAndProb.merge(winOrNot, on='runner_id')
        payout = placeBets(classAndProb)
        payouts = np.append(payouts, payout)
        
    print('mean payout: {}, std payout: {}'.format(np.mean(payouts), np.std(payouts)))
    