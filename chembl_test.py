#TEST WITH CHEMBL database
#from __future__ import division
import numpy as np
import pandas as pd
import scipy.io
import random
import macau

#Test Parameters
test_prop=0.1
latents=20
alpha=5.0
burn=400
samples=1600

df=pd.read_csv("./outputs/chembl_23/chembl_23_pic50_cens.csv").dropna()

#df["test"]=0
#df.loc[df["cens"]==0].sample(frac=0.9)["test"]=1
df.at[random.sample(df.loc[df["cens"]==0].index,int(test_prop*df.shape[0])),"test"]=1
print(df.head())

#Test flag
Test=df.pivot(index='cmpd_id',columns='target_id',values='test').fillna(0).astype(bool).as_matrix()

#Activations
#D=df[["cmpd_id","target_id","pic50_cens"]]

#print(D.head())
Dmat=df.pivot(index='cmpd_id',columns='target_id',values='pic50_cens').fillna(0).as_matrix()
#print(D.loc[D["pic50_cens"]==0])
Dtest=np.zeros(Dmat.shape)
Dtest[Test]=Dmat[Test]
#Dtest=Dmat[Test].fillna(0)
Dmat[Test]=0


#Make sparse
Y=scipy.sparse.coo_matrix(Dmat)
Ytest=scipy.sparse.coo_matrix(Dtest)
print("---------")


#censoring matrix
Cens=df[["cmpd_id","target_id","cens"]]
Cmat=Cens.pivot(index='cmpd_id',columns='target_id',values='cens').fillna(0)
C=scipy.sparse.coo_matrix(Cmat)


#
Dmat[Cmat.astype(bool).as_matrix()]=0
Y_nocens=scipy.sparse.coo_matrix(Dmat)


#Side information
ecfp=scipy.io.mmread("./outputs/chembl_23/ecfp.mtx")
result=macau.macau(Y=Y,Ytest=Ytest,side=[ecfp,None], num_latent=latents,precision=alpha,burnin=burn,nsamples=samples)
result2=macau.macau(Y=Y_nocens,Ytest=Ytest,side=[ecfp,None],num_latent=latents,precision=alpha,burnin=burn,nsamples=samples)

print(result)
print("-----------")
print(result2)
