from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

from matminer.featurizers.bandstructure import BandFeaturizer
from matminer.featurizers.dos import DOSFeaturizer

from tqdm import tqdm
import pandas as pd

def apply_featurizers(criterion, properties, mpdr):
    df = mpdr.get_dataframe(criteria=criterion, properties=properties)
    df = BandFeaturizer().featurize_dataframe(df, col_id="bandstructure",ignore_errors=True)
    df =  DOSFeaturizer().featurize_dataframe(df, col_id="dos",ignore_errors=True)
    return df.drop(["bandstructure", "dos"], axis=1)

def featurize_by_material_id(material_ids, api_key, fileName = False, props=None):
    """
    LOW MEMORY DEMAND function (compared to matminer, pymatgen), but without
    returning "bandstructure"- and "dos"- objects as features.
    Args
    ----------
    material_ids : list
        List containing strings of materials project IDs
    api_key : string
        Individual API-key for Materials Project.

    fileName : str
        Path to file, e.g. "data/aflow_ml.csv"
        Writing to a file during iterations. Recommended for large entries.
    props : list (Cannot contain "band_gap")
        Containing the wanted properties from Materials Project

    Returns
    -------
    Pandas DataFrame
        A DataFrame containing "bandstructure"- and "dos"-featurized features,
        in addition to features given in props.
    """


    mpd = MPDataRetrieval(api_key)

    if props == None:
        properties = ["material_id","full_formula", "bandstructure", "dos"]

    elif props != None:
        properties=props

    firstIndex=True

    for i, mpid in tqdm(enumerate(material_ids)):
        criteria = {"task_id":{"$eq": mpid}}
        if firstIndex:
            FeaturizedEntries = apply_featurizers(criteria, properties, mpd)
            firstIndex = False
            continue
        currentIterationBandStructures = apply_featurizers(criteria, properties, mpd)
        FeaturizedEntries = pd.concat([FeaturizedEntries,currentIterationBandStructures])
        if (fileName) and (i % 50 == 0):
            FeaturizedEntries.to_csv(fileName, sep=",")
    if fileName:
        FeaturizedEntries.to_csv(fileName, sep=",")
    return FeaturizedEntries

def runFeaturizeMP():
    api_key = "b7RtVfJTsUg6TK8E"
    MP_entries = pd.read_csv("data/data_extraction/entire_MP_data/MP/MP.csv", sep=",")
    print(MP_entries.shape)
    fileName = "data/data_extraction/entire_MP_data/MP/MP_featurized.csv"
    featurize_by_material_id(MP_entries["material_id"].values, api_key, fileName)

def reQueueMPFeaturizer():
    #reading entries from MP
    MP_entries =    pd.read_csv("data/data_extraction/entire_MP_data/MP/MP.csv", sep=",")
    MP_featurized = pd.read_csv("data/data_extraction/entire_MP_data/MP/MP_featurized.csv", sep=",")

    howFar = MP_entries[MP_entries["material_id"] == MP_featurized["material_id"].iloc[-1]].index.values
    MP_featurized.to_csv("data/data_extraction/entire_MP_data/MP/MP_featurized_" + str(howFar[0]) + ".csv", sep=",", index=False)

    api_key = "b7RtVfJTsUg6TK8E"
    fileName = "data/data_extraction/entire_MP_data/MP/MP_featurized.csv"
    featurize_by_material_id(MP_entries["material_id"].iloc[howFar[0]+1:].values, api_key, fileName)

if __name__ == '__main__':
    #runFeaturizeMP()
    reQueueMPFeaturizer()

    #print(FeaturizedEntries)
