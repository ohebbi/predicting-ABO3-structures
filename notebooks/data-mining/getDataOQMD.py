import qmpy_rester as qr
import pandas as pd
## Return list of data
with qr.QMPYRester() as q:
    kwargs = {
        #‘elements’: ‘Fe,Mn’,                    # include element Fe and Mn
        #‘nelements’: ‘<5’,                      # less than 4 element species in the compound
        #‘_oqmd_stability’: ‘<0’,                # stability calculted by oqmd is less than 0
        "_oqmd_band_gap": ">0.5",
        "limit": "65000"
        }
    list_of_data = q.get_optimade_structures(**kwargs)

oqmd = pd.DataFrame(list_of_data["data"])

print(list_of_data)
oqmd.to_csv("data/OQMD_DATA.csv", sep=",", index = False)
