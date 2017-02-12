import pandas as pd
import numpy as np



def addOdds(classAndProb, all_odds):
    df_odds = all_odds[all_odds['runner_id'].isin(classAndProb.runner_id)]
    average_win_odds = df_odds.groupby(['runner_id'])['odds_one_win'].mean()
    
    #if there are no odds for a horse, we're going to drop it.
    average_win_odds[average_win_odds == 0] = np.nan
    average_win_odds = pd.DataFrame(average_win_odds).reset_index().rename(columns={'odds_one_win':'odds'})
    average_win_odds = average_win_odds.dropna(subset=['odds'])

    return classAndProb.merge(average_win_odds, on='runner_id')


def trainTestSplit(data, market_ids):
    data = data.sort_values(by='market_id')
    numRaces = data.shape[0]
    train_end = np.floor(0.7 * num_races)
    train = data.iloc[:train_end, :]
    test = data.iloc[:train_end, :]
    sorted_ids = np.sort(market_ids)
    train_ids = sorted_ids[:train_end]
    test_ids = sorted_ids[train_end:]
    

def fitModel(train, test,  est=None, test_market_ids=None, runnersClasses=None):
    
    print('fitting estimator')
    est.fit(train[:, :-1], train[:, -1])    
    
    print('making predictions')
    probs = est.predict_proba(test[:, :-1])
    
     
    colsToAdd = list(np.arange(probs.shape[1])) # how many classes did we make predictions for
    probs = np.hstack((probs, test_market_ids[:, np.newaxis]))
    colsToAdd.append('market_id')
    
    #now we have a df with each race as a row and each column the prob. an individual horse winning
    probs_df = pd.DataFrame(probs, columns=colsToAdd) 
    
    #reshape that data frame where each row corresponds to a horse, its probability and its race
    probs_re = probs_df.set_index('market_id').stack().reset_index().rename(columns={0:'prob', 'level_1':'class'})

    probs_re['class'] = probs_re['class'].astype(float)
    
    #merge in the runner_id for each horse, so we can later use it to get the horse's odds.
    classAndProb = probs_re.merge(runnersClasses, on=['market_id', 'class'])    
    return classAndProb

def placeBets(df):
    df['placeBet'] = df['prob'] > (1/df['odds'])
    investment = df['placeBet'].sum()
    df['wonBet'] = df['placeBet'] & df['win']
    payout = df[df['wonBet'] == True].odds.sum()
    investment_return = round((payout - investment)/investment*100,2)
    
    print('Total bets: {}'.format(df['placeBet'].sum()))
    print('Total wins: {}'.format(df['wonBet'].sum()))
    return investment_return


def runRaces(data=None, est=None, allOdds=None, n_trials=1):
    from shapeRaces import makeRaces
    from sklearn.cross_validation import train_test_split
    data = data.reindex(np.random.permutation(data.index))

    print('reshaping data')
    raceData, runnerClassDf, market_ids = makeRaces(data)
    winOrNot = data[['win', 'runner_id']]
    bestPayout = -9999
    probs = []
    bestEst = None
    payouts = np.array([])
    for i in range(n_trials):
        print('trial {}'.format(i))
        #train, test, train_market_ids, test_market_ids = trainTestSplit(raceData, market_ids )
        train, test, train_market_ids, test_market_ids = train_test_split(raceData, market_ids)
        classAndProb = fitModel(train, 
                                test, 
                                est=est, 
                                test_market_ids=test_market_ids, 
                                runnersClasses=runnerClassDf)
        
        classAndProb = addOdds(classAndProb, allOdds)
        classAndProb = classAndProb.merge(winOrNot, on='runner_id')
        payout = placeBets(classAndProb)
        payouts = np.append(payouts, payout)
        probs.append(classAndProb)
        if payout > bestPayout:
            bestEst = est
      
    print('mean payout: {}, std payout: {}'.format(np.mean(payouts), np.std(payouts)))
    return payouts, probs, bestEst