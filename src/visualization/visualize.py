import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy(models, names):
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


def plot_important_features(models, names,X, k, n):
    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text='Features used in model (Nruns = {})'.format(k*n)),
                yaxis=dict(title="Number times"),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values, y=model['importantKeys']))

    fig.show()

    fig = go.Figure(
            layout = go.Layout (
                title=go.layout.Title(text="Feature Importance for the 100th iteration".format(k*n)),
                yaxis=dict(title='Relative importance'),
                barmode='group'
            )
        )

    for i, model in enumerate(models):
        fig.add_traces(go.Bar(name=names[i], x=X.columns.values, y=model['relativeImportance']))

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
        mat = confusion_matrix(y.values.reshape(-1,), model["y_pred_full"])
        print(mat)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
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
