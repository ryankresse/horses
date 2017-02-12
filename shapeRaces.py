import pandas as pd
import numpy as np

def shapeToRaces(race):
    
    won = race[race.win == True]
    if won.empty:
        # some races don't have a winner. we'll use the -8888 to remove them later
        wonClass = -8888; 
    else:
        won = won.index.values[0]
        wonClass = race.index.get_loc(won) # get what class the winning horse will be    race =     
    
    market_id = race.iloc[0,:].market_id
    runner_ids = race.runner_id.values    
        
    #drop unneccesary columns before we go to straight numpy arrays
    race = race.drop(['horse_id', 'win','position', 'runner_id', 'market_id'], axis=1)
    #stack horse from the race into one flat row
    st = race.stack().to_frame().T #stack it
        
    #now lose the multi-index and extract just values as just a numpy array
    st.columns = ['{}_{}'.format(col[1], col[0]) for col in st.columns]
    vals = st.values
    
    #add fake columns to create features for 25 horses
    colsToAdd = (race.shape[1] * 25) - st.shape[1]
    ones = (np.ones(colsToAdd) * -999)[:, np.newaxis].T
    withAdded = np.concatenate((vals, ones ), axis=1)
        
    #append label and other meta_data to the row
    withAdded = np.append(withAdded, wonClass)
    
    return (withAdded, runner_ids)

def getMarketList(race):
    #creates a tuple (runnerid, class, and market_id) will make
    # which we'll create a dataframe from.
    ids = np.ones(race.runners.shape[0]) * race.market_id
    classes = np.arange(race.runners.shape[0])
    return list(zip(race.runners, classes, ids))

def getClassDf(races):
    #this method creates a dataframe with each row containing a runner_id,
    #the corresponding class it will have in the predictions, and the market_id
    #we'll need this so that we can match the win probabilities with the horse
    #the correct horse after fitting the model
    runner_ids = races.apply(lambda x: x[1])
    d = pd.DataFrame(runner_ids.reset_index().rename(columns={0: 'runners'}))
    d['market_list'] = d.apply(getMarketList,axis=1)
    runnersClasses = np.vstack(d.market_list.values)
    classDf = pd.DataFrame({'runner_id':runnersClasses[:,0],
                            'class':runnersClasses[:,1],
                            'market_id':runnersClasses[:,2]})
    return classDf
    


def makeRaces(df):
    races = df.groupby('market_id').apply(shapeToRaces)
    vals = races.apply(lambda x: x[0]).values # the data we'll use to train the model
    vals = np.vstack(vals)
    #some races didn't have a winner (denoted by -8888), we remove them
    valsMask = vals[:, -1] > -8888
    vals = vals[valsMask]
    
    market_ids = races.index.values
    #only want market ids of races that hadd a winner
    market_ids = market_ids[valsMask]
    classDf = getClassDf(races)
    return (vals, classDf, market_ids)
       