# Predicting ABO3 structures [![DOI](https://zenodo.org/badge/289915727.svg)](https://zenodo.org/badge/latestdoi/289915727)

This repository contains the main work regarding the validation part of my master
thesis. The work can be found in the notebooks, and is written as a Python package
for complete reproduction with the respective requirements of packages.

## Development

Clone the project, and run "pip install -e ." and you are ready for development.  

## Run project

The application of this project is centered around Jupyter notebooks. It is not neccessary to run anything to see result, only consult the notebooks either here on Github or [Jupyers` nbviewer project](https://nbviewer.jupyter.org/).Project Organization

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── summary        <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │   ├── 01-preprocessing-notebook.ipynb
    │   ├── 02-generateTestData-notebook.ipynb
    │   ├── 03-supervisedModels-notebook.ipynb
    │   └── 04-meshgridAnalysis-notebook.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py



## Update requirements
> virtualenv -p python3 envname

> pip3 install -e .

> pip3 freeze > requirements.txt

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
