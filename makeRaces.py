import pandas as pd
import numpy as np


def shapeToRaces(df):
    
    groups = df.groupby('market_id')
    numGroups = len(groups)
    shape = (numGroups, ((df.shape[1] - 5) * 25) + 3)
    
    #initialize the objects we will return
    #the first needs to be the shape of the numpy array we'll return (num_races, num_features + label + market_id + horse that won)  
    raceData = np.empty(shape)
    runnerClassDf = pd.DataFrame(columns=['market_id', 'class', 'runner_id', 'horse_id'])


    for (market_id, race), i in zip(groups, np.arange(numGroups)):
    
        if i % 500 == 0:
            print('{} out of {}'.format(i, numGroups))
        
        #we need to get the class of the horse that won. However, some races don't seem to have a winner..
        #so we just skip those
        won = race[race.win == True]
        if won.empty:
            continue;
        won = won.index.values[0]
        wonClass = race.index.get_loc(won) # get what class the winning horse will be    race =     
    
        #we'll use this data to so that we can match probability to horse after we train the model
        #and make predictions
        d = race[['horse_id', 'runner_id', 'market_id']]
        d.loc[:,'class'] = np.arange(d.shape[0])
        runnerClassDf = pd.concat([runnerClassDf, d])

        
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
        withAdded = np.append(withAdded, won)
        withAdded = np.append(withAdded, wonClass)
        withAdded = np.append(withAdded, market_id)
        
        #fill the object we'll eventually return with that race's data
        raceData[i, :] = withAdded
    
    #initially, we created an empty row for every race in our data,
    #but we filtered out races that had no winners. As it stands, we still have
    #empty rows for them, so we need to remove them
    zeroMask = (raceData[:,0:] < 1)[:,0]
    raceData = raceData[~zeroMask, :]
        
    return (raceData, runnerClassDf)