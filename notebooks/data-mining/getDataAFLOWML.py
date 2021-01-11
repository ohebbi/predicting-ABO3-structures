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
    os.makedirs('data')
    
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
    def writeToFile(fileNamePath, row):
        row.to_csv(fileNamePath,
            sep=",",
            index=False,
            header=False,
            mode='a')
        
    firstIteration = True
    for index, entry in tqdm(entries.iterrows()):

        struc = CifParser.from_string(entry["cif"]).get_structures()[0]
   
        poscar = Poscar(structure=struc)

        ml = AFLOWmlAPI()
        prediction = ml.get_prediction(poscar, 'plmf')

        if firstIteration:
            aflowml_dict = {k: [] for k in prediction.keys()}
            aflowml_dict["full_formula"]    = []
            aflowml_dict["material_id"] = []
            firstIteration = False

        for key in prediction.keys():
            aflowml_dict[key].append(prediction[key])

        aflowml_dict["full_formula"].append(entry["full_formula"])
        aflowml_dict["material_id"].append(entry["material_id"])
        
        if (pathAndFileName):
            writeToFile(fileName, row=pd.DataFrame.from_dict(aflowml_dict).loc[[index]])

    return aflowml_dict

def get_dataframe_AFLOW(entries, fileName = None):
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
    Pandas DataFrame
        A DataFrame containing features as compound and material id,
        as well as the keys in the AFLOW-ML algorithm Property
        Labeled Material Fragments.
    """
    return pd.DataFrame.from_dict(get_dataframe_AFLOWML(entries, fileName))

if __name__ == '__main__':

    #reading entries from MP
    MP_entries = pd.read_csv("data/MP_data_stage_2.csv", sep=",")

    AFLOW_ML = get_dataframe_AFLOWML(entries=MP_entries, pathAndFileName="data/AFLOWML_data_temp.csv")

    #writing to file
    AFLOW_ML.to_csv("data/AFLOWML_data.csv", sep=",", index = False)
