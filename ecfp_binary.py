
import pandas as pd
import scipy.io
import numpy as np

df=pd.read_csv("./outputs/chembl_23/chembl_23/chembl_23_ecfp.csv") #ECFP FEATURES FOR ALL DRUGS
df_d=pd.read_csv("./outputs/chembl_23/chembl_23/chembl_23_pic50_cens.csv") #PIC50 SCORES FOR SPECIFIC COMBINATIONS

#Prune the dataframe from the unfrequent features.
freq_threshold=4 # if a feature is present in less compounds than this threshold, we remove it.
count_feats=df.groupby("feature")["compound"].nunique()
idx_taken=count_feats[count_feats>freq_threshold].index
df=df[df["feature"].isin(idx_taken)]

#Restict ECFP DF to drugs of interest only.
df=df[df["compound"].isin(df_d["cmpd_id"].unique())]

#mapping between feature and its "binary" position
old_map=df["feature"].unique()
new_map=range(0,len(old_map))
map_dict=dict(zip(old_map,new_map))

#Mapping between each compound and its row number.
new_dict=dict(zip(sorted(df_d["cmpd_id"].unique()),df_d.index.values))

#Non-zero columns ecfp for each compound
cols_idx=df["feature"].map(map_dict).values
rows_idx=df["compound"].map(new_dict).values

ecfp=scipy.sparse.coo_matrix((np.ones(cols_idx.shape),(rows_idx,cols_idx)))

scipy.io.mmwrite("./outputs/chembl_23/chembl_23/ecfp",ecfp)
