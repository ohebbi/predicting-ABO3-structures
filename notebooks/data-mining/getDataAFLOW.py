#!/usr/bin/env python3

# Tables
import pandas as pd

# Visual representation of the query process
from tqdm import tqdm

# Query library
from aflow import *

import os
if not os.path.exists('data'):
    os.makedirs('data')
    
def get_dataframe_AFLOW(compound_list, keys, batch_size):
    """
    A function used to make a query to AFLOW. 
    ...
    Args
    ----------
    compound_list : list (dim:N)
        A list of strings containing full formula, eg. H2O1 or Si1C1
    keys : list (dim:M)
        A list containing the features of the compound one wants to extract,
        eg. Egap
    batch_size : int
        Number of data entries to return per HTTP request

    Returns
    -------
    pd.DataFrame (dim:MxN)
        A DataFrame containing the resulting matching queries. This can result
        in several matching compounds
    """
    index = 0
    # Looping through every compound
    for compound in tqdm(compound_list):
        print("Current query: {}".format(compound))

        # Original filter. Here we choose entries in AFLOW based on ICSD.
        results = search(catalog='icsd', batch_size=batch_size)\
            .filter(K.compound==compound)

        # If search returns matching query
        if len(results)>0:
            # Looping through every result gained from query with a compound
            for result in tqdm(results):
                for key in keys:
                    try:
                        # The query for features
                        aflow_dict[key].append(getattr(result,key))
                    except:
                        # If no matching feature for given compound
                        aflow_dict[key].append("None")

                pd.DataFrame.from_dict(aflow_dict).loc[[index]].to_csv("data/AFLOW_DATA_temp.csv",
                sep=",",
                index=False,
                header=False,
                mode='a')
                index += 1
        else:
            print("No compound is matching the search")
            continue

    return pd.DataFrame.from_dict(aflow_dict)

if __name__ == '__main__':

    #reading entries from MP
    MP_entries = pd.read_csv("data/MP_DATA.csv", sep=",")
    compound_list = list(MP_entries['full_formula'])

    #choosing keys used in AFLOW. We will here use all features in AFLOW.
    keys = list(pd.read_csv("data/keywords_AFLOW.txt", sep=",").columns)
    aflow_dict = {k: [] for k in keys}

    AFLOW_entries = get_dataframe_AFLOW(compound_list=compound_list, keys=keys, batch_size=1000)

    #writing to csv
    AFLOW_entries.to_csv("data/AFLOW_DATA.csv", sep=",", index = False)
