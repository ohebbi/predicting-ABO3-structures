from pymatgen import MPRester
import pandas as pd

with MPRester("b7RtVfJTsUg6TK8E") as mpr:

        criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                    "band_gap": {"$gt": 0.1}
                    }

        props = ["material_id","full_formula","icsd_ids", "spacegroup.number","band_gap","run_type","cif","e_above_hull", "elements", "structure"]#,'pretty_formula','e_above_hull',"band_gap"]
        #entries = mpd.get_dataframe(criteria=criteria, properties=props)
        entries = pd.DataFrame(mpr.query(criteria=criteria, properties=props))
        entries.to_csv("MP.csv", sep=",", index = False)
