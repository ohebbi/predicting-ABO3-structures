#Feature selections
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, make_scorer, f1_score

#Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def chooseSampler(sampleMethod):
    if sampleMethod == "under":
        return ("underSampler", RandomUnderSampler(sampling_strategy="majority"))

    elif sampleMethod == "over":
        return ("overSampler", SMOTE(sampling_strategy="minority"))

    elif sampleMethod == "both":
        return "overSampler", SMOTE(sampling_strategy="minority"),\
               "underSampler", RandomUnderSampler(sampling_strategy="majority")

    else:
        return None

def getPipe(model, sampleMethod):
    sampler = chooseSampler(sampleMethod)
    if not (sampler):
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', model)
        ])

    if len(sampler)==2:
        return Pipeline([
            ('scale', StandardScaler()),
            ('model', model)
        ])

    elif len(sampler)==4:
        return Pipeline([
            ('scale', StandardScaler()),
            sampler[0:2],
            sampler[2:4],
            ('model', model)
        ])

    else:
        raise ValueError("Wrong number of samplers: len(sampler)={}".format(len(sampler)))

def findParamGrid(model):
    typeModel = type(model)

    if typeModel == type(RandomForestClassifier()):
        return {"model__n_estimators": [10,50,100,200,500,1000],
                "model__max_features": ['auto', 'sqrt', 'log2'],
                "model__max_depth" : np.arange(1,8),
                "model__criterion" :['gini'],#, 'entropy'],
                }
    elif typeModel == type(GradientBoostingClassifier()):
        return {"model__loss":["deviance"],
                #"model__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "model__min_samples_split": np.linspace(0.1, 0.5, 3),
                "model__min_samples_leaf": np.linspace(0.1, 0.5, 3),
                "model__max_depth":np.arange(1,8),
                "model__max_features":["log2","sqrt"],
                #"model__criterion": ["friedman_mse",  "mae"],
                #"model__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                "model__n_estimators":[10,50,100,200,500,1000],
                }
    elif typeModel == type(LogisticRegression()):#penalty{‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
        return {"model__penalty":["l2"],# "l2", "elasticnet", "none"],
                "model__C": np.logspace(-3,5,7),
                "model__max_iter":[200, 400, 600, 800],
                }
    else:
        raise TypeError("No model has been specified: type(model):{}".format(typeModel))


def applyGridSearch(X, y, model, cv, sampleMethod="under"):
    param_grid = findParamGrid(model)

    ## TODO: Insert these somehow in gridsearch (scoring=scoring,refit=False)
    scoring = {'accuracy':  make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall':    make_scorer(recall_score),
               'f1':        make_scorer(f1_score),
               }

    # Making a pipeline
    pipe = getPipe(model, sampleMethod)

    # Do a gridSearch
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, refit="f1",
                        cv=cv,verbose=2,return_train_score=True, n_jobs=-1)
    grid.fit(X, y)

    #print (grid.best_params_)

    return grid.best_estimator_, grid
