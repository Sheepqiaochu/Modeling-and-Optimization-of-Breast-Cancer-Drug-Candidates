import pandas as pd
import numpy as np

MD=pd.read_excel('./data/Molecular_Descriptor.xlsx',header=0)
rows,cols=MD.shape
header=list(MD.columns)
drop=[]
#tmp=0
for i in range(1,cols):
    cnt=0
    for j in range(rows):
        if(MD.iloc[j,i]==0):
            cnt=cnt+1
    if(cnt/rows>0.9):
        drop.append(header[i])
        #tmp=tmp+1
#print(tmp)
MD=MD.drop(drop,axis=1)
#MD.to_csv('./data/MD_train_dropzero_mval.csv',header=True,index=False)

rows,cols=MD.shape
header=list(MD.columns)
drop=[]
#tmp=0
for i in range(1,cols):
    cnt=0
    mean=np.mean(MD.iloc[:,i])
    std=np.std(MD.iloc[:,i])
    up=mean+3*std
    bottom=mean-3*std
    for j in range(rows):
        if(MD.iloc[j,i]>up):
            MD.iloc[j,i]=up
            cnt=cnt+1
        if(MD.iloc[j,i]<bottom):
            MD.iloc[j,i]=bottom
            cnt=cnt+1
    if(cnt>100):
        drop.append(header[i])
        #tmp=tmp+1
#print(tmp)
MD=MD.drop(drop,axis=1)
#MD.to_excel('./data/MD_train_dropzero.xlsx',header=True,index=False)
MD.to_csv('./data/MD_train_dropzero.csv',header=True,index=False)