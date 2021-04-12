
from pathlib import Path
import joblib

def runPredictions(classifier, trainingData, trainingTarget, testData, prettyName:str , cubicCase:bool = False):
    print(classifier)
    print(trainingData.shape)
    #train the model
    classifier.fit(trainingData, trainingTarget)

    if cubicCase:
        label = "perovskite"
    else:
        label = "cubic-perovskite"

    file_path = Path(Path.cwd().parent / "models" / label / Path(prettyName + ".pkl"))

    with file_path.open('wb') as fp:
        joblib.dump(classifier, fp)

    file_path = Path(Path.cwd().parent / "models" / label / Path(prettyName + "features.pkl"))

    with file_path.open('wb') as fp:
        joblib.dump(trainingData.columns, fp)
    #predict
    predictions = classifier.predict(testData)
    probability = classifier.predict_proba(testData)

    #the predicted perovskites
    return predictions, probability[:,1]
