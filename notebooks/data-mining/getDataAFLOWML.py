#!/usr/bin/env python3

# Tables
import pandas as pd

# Visual representation of the query process
from tqdm import tqdm

# ML library and structural library
from aflowml.client import AFLOWmlAPI
from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar

def get_dataframe_AFLOW_ML(entries):
    """
    A function used to make a query to AFLOW, and will return as much
    information as possible.
    ...
    Parameters
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

    Returns
    -------
    Pandas DataFrame
        A DataFrame containing features as compound and material id,
        as well as the keys in the AFLOW-ML algorithm Property
        Labeled Material Fragments.
    """

    firstIteration = True
    for index, entry in tqdm(entries.iterrows()):

        outF = open("data/prepForPoscar.cif", "w")
        outF.writelines(entry["cif"])
        outF.close()

        struc = Structure.from_file("data/prepForPoscar.cif")
        poscar = Poscar(structure=struc)

        ml = AFLOWmlAPI()
        prediction = ml.get_prediction(poscar, 'plmf')

        if firstIteration:
            AFLOW_ML = {k: [] for k in prediction.keys()}
            AFLOW_ML["full_formula"]    = []
            AFLOW_ML["material_id"] = []
            firstIteration = False

        for key in prediction.keys():
            AFLOW_ML[key].append(prediction[key])

        AFLOW_ML["full_formula"].append(entry["full_formula"])
        AFLOW_ML["material_id"].append(entry["material_id"])

        pd.DataFrame.from_dict(AFLOW_ML).loc[[index]].to_csv("data/AFLOW_ML_DATA_temp.csv",
        sep=",",
        index=False,
        header=False,
        mode='a')

    return pd.DataFrame.from_dict(AFLOW_ML)

if __name__ == '__main__':

    #reading entries from MP
    MP_entries = pd.read_csv("data/MP_DATA.csv", sep=",")

    AFLOW_ML = get_dataframe_AFLOW_ML(entries=MP_entries)

    #writing to
    AFLOW_ML.to_csv("data/AFLOW_ML_DATA.csv", sep=",", index = False)
