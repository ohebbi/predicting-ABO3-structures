import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from typing import Optional

import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import auc, average_precision_score, roc_curve, precision_recall_curve, f1_score
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# textwidth in LateX
width = 411.14224

height = ((5**.5 - 1) / 2 )*width

width_plotly = 548.1896533333334 #pt to px
height_plotly = ((5**.5 - 1.) / 2 )*width_plotly
tex_fonts = {
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.labelsize":12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize":12
}
plt.rcParams.update(tex_fonts)

def set_size(width, fraction=1, subplots=(1,1)):
    """ Set fgure dimensions to avoid scaling in LateX.

    Args
    ---------
    width : float
            Document textwidth or columnwidth in pts
    fraction : float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and column of subplots
    Returns
    ---------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1/72.27

    # Golden ratio to set aeshetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def plot_accuracy(models, names, xlabel = "Cross validation folds"):

    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=set_size(width, 0.75, subplots=(3,1)))
    for i, model in enumerate(models):
        ax1.plot(model['trainAccuracy'], label=names[i])#, color = color[j])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.set_title('Training accuracy')
    ax1.legend(loc='best')

    for i, model in enumerate(models):
        ax2.plot(model['testAccuracy'], label=names[i])#, color = color[j])
    ax2.xaxis.set_major_formatter(plt.NullFormatter())
    ax2.set_title('Test accuracy')

    for i, model in enumerate(models):
        ax3.plot(model['f1_score'], label=names[i])#, color = color[j])
    ax3.set_title('f1-score')

    ax3.set_xlabel(xlabel)
    fig.tight_layout()

    plt.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['trainAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Train accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['testAccuracy'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Test accuracy',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Accuracy')
    fig.show()

    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['std'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Standard deviation up to every iteration',
                   xaxis_title='Cross validation folds',
                   yaxis_title='std')
    fig.show()


    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Scatter(y=model['numPredPero'],
                    mode='lines',
                    name=names[i]))
    fig.update_layout(title='Number of predicted perovskites',
                   xaxis_title='Cross validation folds',
                   yaxis_title='Number perovskites')
    fig.show()


def plot_important_features(models, names, X, k, n, fileName):
    fig = make_subplots(rows=models.shape[0], cols=1, shared_xaxes=True)
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                           "paper_bgcolor": "rgba(0, 0, 0, 0)",
                          },
                        barmode='group',
                        autosize=False,
                        width=width_plotly,
                        height=height_plotly,
                        margin=dict(l=0, r=0, t=25, b=0),
                        title=go.layout.Title(text="Mean feature importance for {} iterations".format(k*n)),
                        #xaxis=dict(title=xlabel),
                        #yaxis=dict(title="Relative importance"),
                        font=dict(family="Palatino",
                                  color="Black",
                                  size=12),)

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values,
                    y=np.mean(model["relativeImportance"], axis=0),
                    error_y=dict(type='data', array=np.std(model["relativeImportance"], axis=0))), cols = 1, rows=i+1)

    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                    "reports" / "figures"
                                    / Path(fileName)))
    fig.show()


def plot_confusion_metrics(models, names, data,  k, n, abbreviations=[], cubicCase=False):
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="False positives (Nruns = {})".format(k*n)),
                yaxis=dict(title='Counts'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        if cubicCase is not False:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data[abbreviations[i]]['Compound'][model['falsePositives'] > 0],
                                  y=model['falsePositives'][model['falsePositives'] > 0]))
        else:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data['Compound'][model['falsePositives'] > 0],
                                  y=model['falsePositives'][model['falsePositives'] > 0]))

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="False negatives (Nruns = {})".format(k*n)),
                yaxis=dict(title='Counts'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        if cubicCase is not False:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data[abbreviations[i]]['Compound'][model['falseNegatives'] > 0],
                                  y=model['falseNegatives'][model['falseNegatives'] > 0]))
        else:
            fig.add_traces(go.Bar(name=names[i],
                                  x=data['Compound'][model['falseNegatives'] > 0],
                                  y=model['falseNegatives'][model['falseNegatives'] > 0]))

    fig.show()

def plot_confusion_matrix(models, y, data, abbreviations, names, k, n, cubicCase = False):
    confidence = np.linspace(0,k*n,k*n)
    #confidence = 95 # % confidence. Put as 50 or more.
    mat = np.zeros((len(confidence),2,2))
    bigger_than = 100-confidence

    #Finding true positives and true negatives as a function of confidence
    for j, model in enumerate(models):

        for i, conf in enumerate(confidence):
            bigger_than = 100 - conf
            confidence[i] = bigger_than

            if cubicCase:
                model["y_pred_full"] =  y[abbreviations[j]]["Cubic"].values.reshape(-1,).copy()

                model["y_pred_full"]\
                    [data[abbreviations[j]]['Compound'][model['falseNegatives'] > bigger_than].index] = 0

                model["y_pred_full"]\
                    [data[abbreviations[j]]['Compound'][model['falsePositives'] > bigger_than].index] = 1

                mat[i] = confusion_matrix(y[abbreviations[j]]["Cubic"].values.reshape(-1,), model["y_pred_full"])

            else:
                model["y_pred_full"] =  y.values.reshape(-1,).copy()
                model["y_pred_full"]\
                    [data['Compound'][model['falseNegatives'] > bigger_than].index] = 0

                model["y_pred_full"]\
                    [data['Compound'][model['falsePositives'] > bigger_than].index] = 1

                mat[i] = confusion_matrix(y.values.reshape(-1,), model["y_pred_full"])

        plt.plot(confidence, mat[:,0,1])
        plt.plot(confidence, mat[:,1,0])
        plt.plot([50,50],[-2,np.max(mat[:,0,1])], "--")

        plt.xlabel("Confidence / counts of wrongly predictions")
        plt.ylabel("Number of compounds")
        plt.title("Confusion matrix for predictions 100 times {}".format(names[j]))
        plt.legend(["False negatives", "False positives", "The article¨s chosen C"])
        plt.show()

def confusion_matrix_plot(models, y, names):
    #print(mat)
    for i, model in enumerate(models):

        sns.heatmap(model["confusionMatrix"], square=True, annot=True, fmt='d', cbar=False,
                xticklabels=[0,1],
                yticklabels=[0,1])
        plt.xlabel('true label')
        plt.ylabel('predicted label');
        plt.title("Confusion matrix {}".format(names[i]))
        plt.show()

def findCorrectlyPredictedPerovskites(model, data, allowedMiscalcs):
    predictedPerovskites = data.copy()
    remove_indices1 = np.array(data['Compound'][model['falseNegatives'] > allowedMiscalcs].index)
    remove_indices2 = np.array(data['Compound'][model['falsePositives'] > allowedMiscalcs].index)
    remove_indices3 = np.array(data['Compound'][data.Perovskite==-1].index)

    idx = list(set(np.concatenate((remove_indices1,remove_indices2, remove_indices3)))) #only unique indices

    predictedPerovskites = data.drop(idx)
    predictedPerovskites.reset_index(drop=True, inplace=True)
    predictedPerovskites["Cubic"] = np.where(predictedPerovskites["Cubic"] < 0, 0, 1)

    return predictedPerovskites


def runSupervisedModel(classifier,
                       X: pd.DataFrame,
                       y,
                       k: int,
                       n: int,
                       cv,
                       title: str,
                       featureImportance: Optional[bool] = False,
                       resamplingMethod: Optional[str] = "None"):


    modelResults = {
        'trainAccuracy':   np.zeros(n*k),
        'testAccuracy':    np.zeros(n*k),
        'f1_score':        np.zeros(n*k),
        'std':             np.zeros(n*k),
        'importantKeys':   np.zeros(len(X.columns.values)),
        'numPredPero':     np.zeros(n*k),
        'confusionMatrix': np.zeros((len(y), len(y))),
        'falsePositives':  np.zeros(len(y)),
        'falseNegatives':  np.zeros(len(y)),
        'relativeImportance': np.zeros((n*k, len(X.columns.values)))
        }

    # Initializing Creating ROC metrics
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)
    fig2, ax2 = plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))


    #  Initializing precision recall metrics
    fig1, ax1= plt.subplots(1,1, figsize=(set_size(width, 1)[0], set_size(width, 1)[0]))
    y_real = []
    y_proba = []

    # splitting into 50%/50% training and test data if n_splits = 2, or 90%/10% if n_splits=10
    #rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n, random_state=random_state)

    if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
        sel_classifier = SelectFromModel(classifier.named_steps["model"])

    for i, (train_index, test_index) in tqdm(enumerate(cv.split(X, y))):

        #partition the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit the model
        classifier.fit(X_train, y_train)

        if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
            sel_classifier.fit(X_train, y_train)

        #predict on test set
        y_pred      = classifier.predict(X_test)
        probas_     = classifier.predict_proba(X_test)

        #Finding predicted labels on all data based on training data.
        y_pred_full = classifier.predict(X)

        ############################################
        ## Compute ROC curve and area under curve ##
        ############################################

        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax2.plot(fpr, tpr, lw=1, color='grey', alpha=0.4)
                # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        ######################################
        ## Finding precision recall metrics ##
        ######################################

        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])

        # Plotting each individual PR Curve
        ax1.plot(recall, precision, lw=1, alpha=0.3, color='grey')
                 #label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

        y_real.append(y_test)
        y_proba.append(probas_[:, 1])

        ######################################
        ##### Finding FP and FN metrics ######
        ######################################

        falsePositives = np.nonzero(y_pred_full.reshape((-1,)) > y)
        falseNegatives = np.nonzero(y_pred_full.reshape((-1,)) < y)

        #claim the scores
        modelResults['trainAccuracy'][i] = classifier.score(X_train, y_train)
        modelResults['testAccuracy'][i]  = classifier.score(X_test, y_test)
        modelResults['f1_score'][i]      = f1_score(y_test, y_pred)
        modelResults['std'][i]           = np.std(modelResults['testAccuracy'][0:i+1])
        modelResults['numPredPero'][i]   = np.sum(y_pred_full)
        modelResults['confusionMatrix']  = confusion_matrix(y_test, y_pred)
        modelResults['falsePositives'][falsePositives] += 1
        modelResults['falseNegatives'][falseNegatives] += 1

        if (featureImportance) and (type(classifier["model"]) != type(LogisticRegression())):
            #print(sel_classifier.get_support().shape, modelResults['importantKeys'].shape)
            modelResults['importantKeys'][sel_classifier.get_support()] += 1
            modelResults['relativeImportance'][i] = classifier.named_steps["model"].feature_importances_
        elif type(classifier["model"]) == type(LogisticRegression()):
            #print(classifier.named_steps['model'].coef_)
            modelResults['importantKeys'][:] += 1
            modelResults['relativeImportance'][i] = classifier.named_steps['model'].coef_
    ######################################
    ## Finding precision recall metrics ##
    ######################################
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    ax1.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    ax1.set_title("CV-PR Curve " + str(title))
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")

    ax1.legend(loc="lower right")
    fig1.tight_layout()

    ######################################
    ######## ROC CURVE and AOG ###########
    ######################################

    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax2.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("CV-ROC Curve " + str(title))
    ax2.legend(loc="lower right")
    fig2.tight_layout()

    plt.show()

    print ("Mean accuracy:{:0.5f}".format(np.mean(modelResults['testAccuracy'])))
    print ("Standard deviation:{:0.5f}".format(modelResults['std'][-1]))
    print ("f1-score:{:0.5f}".format(modelResults['f1_score'][-1]))

    return modelResults
def plot_parallel_coordinates(data, color, fileName):

    fig = px.parallel_coordinates(data,
                              color=color,
                              color_continuous_scale=px.colors.diverging.Tealrose)
    fig.update_layout(
                        autosize=False,
                        width=width_plotly,
                        height=height_plotly,
                        #title=go.layout.Title(text="Parallel coordinate plot of dataset"),
                        #xaxis=dict(title=xlabel, range=[-0.1,10]),
                        #yaxis=dict(title=ylabel, range=[-0.1,10]),
                        font=dict(family="Palatino",
                                  color="Black",
                                  size=12),)

    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"
                                / Path(fileName)))
    fig.show()

def plot_distribution_histogram(data, fileName):

    fig = px.histogram(data, x="t", color="Perovskite",opacity=0.75, #color_discrete_sequence=px.colors.diverging.Tealrose,# color_discrete_midpoint=0,
                        marginal="box", # or violin, rug
                        hover_data=data.columns)
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",
                           "paper_bgcolor": "rgba(0, 0, 0, 0)",
                          },
                        autosize=False,
                        width=width_plotly,
                        height=height_plotly*0.75,
                        margin=dict(l=0, r=0, t=0, b=0),
                        #title=go.layout.Title(text="Parallel coordinate plot of dataset"),
                        #xaxis=dict(title=xlabel, range=[-0.1,10]),
                        #yaxis=dict(title=ylabel, range=[-0.1,10]),
                        font=dict(family="Palatino",
                                  color="Black",
                                  size=12),)

    fig.write_image(str(Path(__file__).resolve().parents[2] / \
                                "reports" / "figures"
                                / Path(fileName)))
    fig.show()
