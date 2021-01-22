#!/usr/bin/env python3

# Tables
import pandas as pd

# Visual representation of the query process
from tqdm import tqdm

# ML library and structural library
from aflowml.client import AFLOWmlAPI

from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.cif import CifParser

import os
if not os.path.exists('data'):
    os.makedirs('data/data_extraction/AFLOWML')

def get_dataframe_AFLOWML(entries, fileName = False):
    """
    A function used to initialise AFLOW-ML with appropiate inputs.
    ...
    Args
    ----------
    entries : Pandas DataFrame
    {
        "cif": {}
            - Materials Project parameter "cif", which is a dict
        "compound": []
            - list of strings
        "material id": []
            - list of strings
    }
    fileName : str
        Path to file, e.g. "data/aflow_ml.csv"
        Writing to a file during iterations. Recommended for large entries.

    Returns
    -------
    dict
        A dictionary containing features as compound and material id,
        as well as the keys in the AFLOW-ML algorithm Property
        Labeled Material Fragments.
    """

    firstIteration = True
    for index, entry in tqdm(entries.iterrows()):

        struc = CifParser.from_string(entry["cif"]).get_structures()[0]

        poscar = Poscar(structure=struc)

        ml = AFLOWmlAPI()
        prediction = ml.get_prediction(poscar, 'plmf')

        if firstIteration:
            aflowml_dict = {k: [] for k in prediction.keys()}
            aflowml_dict["full_formula"] = []
            aflowml_dict["material_id"]  = []
            firstIteration = False

        for key in prediction.keys():
            aflowml_dict[key].append(prediction[key])

        aflowml_dict["full_formula"].append(entry["full_formula"])
        aflowml_dict["material_id"].append(entry["material_id"])
        if (fileName) and (index % 50 == 0):
            pd.DataFrame.from_dict(aflowml_dict).to_csv(fileName, sep=",",index=False)
    if fileName:
        pd.DataFrame.from_dict(aflowml_dict).to_csv(fileName, sep=",",index=False)

    return aflowml_dict

def get_dataframe_AFLOW(entries, fileName = None):
    """
    A function used to initialise AFLOW-ML with appropiate inputs.
    ...
    Args
    ----------
    See get_dataframe_AFLOW()

    Returns
    -------
    Pandas DataFrame
        A DataFrame containing features as compound and material id,
        as well as the keys in the AFLOW-ML algorithm Property
        Labeled Material Fragments.
    """
    return pd.DataFrame.from_dict(get_dataframe_AFLOWML(entries, fileName))

def dataMiningProcess():
    #reading entries from MP
    MP_entries = pd.read_csv("data/stage_2/MP_data_stage_2.csv", sep=",")

    AFLOW_ML = get_dataframe_AFLOWML(entries=MP_entries, pathAndFileName="data/data_extraction/AFLOWML/AFLOWML_data.csv")

    #writing to file
    AFLOW_ML.to_csv("data/data_extraction/AFLOWML/AFLOWML_data.csv", sep=",", index = False)

def allMPCompounds():
    #reading entries from MP
    MP_entries = pd.read_csv("data/data_extraction/entire_MP_data/MP/MP.csv", sep=",")

    AFLOW_ML = get_dataframe_AFLOWML(entries=MP_entries, fileName="data/data_extraction/entire_MP_data/AFLOWML/AFLOWML_data.csv")

    #writing to file
    AFLOW_ML.to_csv("data/data_extraction/entire_MP_data/AFLOWML/AFLOWML_data.csv", sep=",", index = False)


def reQueueAllMPCompounds():
    #reading entries from MP
    MP_entries = pd.read_csv("data/data_extraction/entire_MP_data/MP/MP.csv", sep=",")
    previous_AFLOWML_entries = pd.read_csv("data/data_extraction/entire_MP_data/AFLOWML/AFLOWML_data.csv", sep=",")
    #print(previous_AFLOWML_entries.shape[0])
    #print(previous_AFLOWML_entries.iloc[-1])
    howFar = MP_entries[MP_entries["full_formula"] == previous_AFLOWML_entries["full_formula"].iloc[-1]].index.values
    previous_AFLOWML_entries.to_csv("data/data_extraction/entire_MP_data/AFLOWML/AFLOWML_data_" + str(howFar[0]) + ".csv", sep=",", index=False)

    #print(MP_entries.iloc[howFar[0]+1:])

    AFLOW_ML = get_dataframe_AFLOWML(entries=MP_entries.iloc[howFar[0]+1:], fileName="data/data_extraction/entire_MP_data/AFLOWML/AFLOWML_data.csv")



if __name__ == '__main__':
    #dataMiningProcess()

    #allMPCompounds()
    reQueueAllMPCompounds()
