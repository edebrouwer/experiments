# Dataset filters:

# 1) Select all assay.assay_organism="Homo sapiens", target_dictionary.targe_type="SINGLE PROTEIN" proteins that have at least N standard_type="IC50" measurements (N=100 or 200)

#TODO: check: Rattus norvegicus, Mus musculus

#TODO: check these: ED50, AC50, EC50, MIC, Ki, Activity, Inhibition, GI50, Potency (later the higher number)

# 2) Filter according to the standard_units="nM" #TODO: ug.mL-1

# 3) Pick the minimum IC50 for all cells (take care of missing, NA, ...)

# 4) Filter credible values: 10^9 > IC50 >= 10^-5  #TODO: tighten #3<->4

# 5) pIC50 = 9 - log10(IC50)

# 6) Refilter proteins that have at least N compounds

# 7) Remove empty rows

# OUT: matrix market data, compound name, protein name lists


import configargparse
import sqlite3
import pandas as pd
import numpy as np
import scipy.io
import os
import logging


p = configargparse.ArgParser(default_config_files=["default.ini"])
p.add('-c', '--config', required=False, is_config_file=True, help='Config file path')
p.add('--sqlite', required=True, type=str, help="ChEMBL sqlite database")

#p.add("--organism", required=True, help="Organisms for protein filtering" )

#p.add("--targettype", required=True, help="Target type for protein filtering")

p.add('--mincmpdcount', required=True, help='Minimal number of compounds required for an assays', type=int)
p.add('--minassaycount', required=True, help='Minimal number of assays required for a compound', type=int)
p.add('--thresholds', required=True, help="Thresholds for classification", type=float, action="append")
p.add('--datadir', required=True, help="Data directory to write to (append prefix)", type=str)
p.add('--prefix', required=True, help="Prefix for the current dataset", type=str)

options = p.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

outdir = options.datadir + "/" + options.prefix

if not os.path.exists(outdir):
    os.makedirs(outdir)

conn = sqlite3.connect(options.sqlite)

logging.info("Querying sqlite database '%s'" % options.sqlite)
# Homo sapiens, Single Protein, IC50, nM
df = pd.read_sql_query("""SELECT molecule_dictionary.chembl_id as cmpd_id, target_dictionary.chembl_id as target_id,
                          CASE activities.standard_type
                            WHEN 'IC50' THEN activities.standard_value
                            WHEN 'ug.mL-1' THEN activities.standard_value / compound_properties.full_mwt * 1E6
                            END ic50,
                          CASE activities.standard_relation
                            WHEN '<'  THEN '<'
                            WHEN '<=' THEN '<'
                            WHEN '='  THEN '='
                            WHEN '>'  THEN '>'
                            WHEN '>='  THEN '>'
                            ELSE 'drop' END relation
                          FROM molecule_dictionary
                          JOIN activities on activities.molregno == molecule_dictionary.molregno
                          JOIN assays on assays.assay_id == activities.assay_id
                          JOIN target_dictionary on target_dictionary.tid == assays.tid
                          JOIN compound_properties on compound_properties.molregno = molecule_dictionary.molregno
                          WHERE target_dictionary.organism='Homo sapiens' AND target_dictionary.target_type='SINGLE PROTEIN' AND
                                activities.standard_type = 'IC50' AND activities.standard_units IN  ('nM','ug.mL-1') and relation != 'drop' and
                                ic50 < 10e9 AND ic50 >= 10e-5 """, conn)
conn.close()
logging.info("Filtering and thresholding activity data")
# Pick the minimum
df = df.groupby(["target_id","cmpd_id"]).min().reset_index()
# at least N compounds per assay
c  = df.groupby("target_id")["cmpd_id"].nunique()
i  = c[c >= options.mincmpdcount].index
df = df[df.target_id.isin(i)]
# at least M assays per compounds
c  = df.groupby("cmpd_id")["target_id"].nunique()
i  = c[c >= options.minassaycount].index
df = df[df.cmpd_id.isin(i)]


df["pic50"] = 9 - np.log10(df["ic50"])
df.to_csv('temporary_continuous.csv')

#Thresholding
value_vars = []
for thr in options.thresholds:
    value_vars.append("%1.1f" % thr)
    thr_str = "%1.1f" % thr
    ## using +1 and -1 for actives and inactives
    df[thr_str] = (df["pic50"] >= thr) * 2.0 - 1.0
    df[thr_str] = np.where(np.logical_and((df["relation"] == '<'), (df['pic50'] < thr)), np.nan, df[thr_str])
    df[thr_str] = np.where(np.logical_and((df["relation"] == '>'), (df['pic50'] > thr)), np.nan, df[thr_str])
    df["pic50_cens"]=np.where(df["relation"]=='<',np.nan,df["pic50"]) #Not considering above censoring.
    df["cens"]=np.where(df["relation"]=='>',1,0)

logging.info("Saving data into '%s'" % outdir)
melted = pd.melt(df, id_vars=['target_id','cmpd_id'], value_vars=value_vars).dropna()
melted.to_csv('%s/%s_thresh.csv' % (outdir, options.prefix), index = False)

dfpic=df[["cmpd_id","target_id","pic50_cens","cens"]]
dfpic=dfpic.dropna()
dfpic.to_csv('%s/%s_pic50_cens.csv' % (outdir, options.prefix),index=False,columns=["cmpd_id","target_id","pic50_cens","cens"])

#Write unique compound IDs
np.savetxt("%s/%s_compounds.csv" % (outdir, options.prefix), melted["cmpd_id"].unique(), fmt="%s")
np.savetxt("%s/%s_targets.csv" % (outdir, options.prefix), melted["target_id"].unique(), fmt="%s")
