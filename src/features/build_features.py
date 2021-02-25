import pandas as pd

def getStratifiedTrainingData(predictedPerovskites, Name):
    cubics    = predictedPerovskites.iloc[predictedPerovskites["Cubic"][predictedPerovskites["Cubic"]==1].index]
    nonCubics = predictedPerovskites.iloc[predictedPerovskites["Cubic"][predictedPerovskites["Cubic"]!=1].index]
    print(Name, ":")
    print("The amount of cubic perovskites entries in the data is {}, with a total percentage of {}"\
          .format(np.sum(cubics["Cubic"]), np.sum(cubics["Cubic"])/len(predictedPerovskites["Cubic"])))

    # The data trained on should be evenly distributed. Here, we are just picking random numbers.
    nonCubicsSubSet = nonCubics.sample(n = int(percentage*len(predictedPerovskites.index)), random_state=random_state)

    #test to make the reader aware of the distribution in the training data
    if (nonCubicsSubSet.shape!=cubics.shape):
        print("Current shape Cubics: {} and nonCubics: {}".format(cubics.shape, nonCubicsSubSet.shape))

    #Combining the subsets
    stratCubicData = pd.concat([cubics, nonCubicsSubSet])
    stratCubicData.reset_index(drop=False, inplace=True)
    return stratCubicData
