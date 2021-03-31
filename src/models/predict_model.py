def runPredictions(classifier, trainingData, trainingTarget, testData):
    print(classifier)

    #train the model
    classifier.fit(trainingData, trainingTarget)

    #predict
    predictions = classifier.predict(testData)
    probability = classifier.predict_proba(testData)

    #the predicted perovskites
    return predictions, probability[:,1]
