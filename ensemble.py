import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def minmax(dataframe, target):
    dataframe[target]=(dataframe[target]-dataframe[target].min())/(dataframe[target].max()-dataframe[target].min())
    return dataframe

target ='smoking'
GEOMEAN = 'N'
i=0

topper_1=pd.read_csv('submission1.csv')
topper_2=pd.read_csv("submission2.csv")
topper_3=pd.read_csv("submission3.csv")
topper_4=pd.read_csv("submission4.csv")
topper_5=pd.read_csv("submission5.csv")

topper_df = topper_1.copy()
topper_df.columns =['id', 'topper_1']
topper_df['topper_1'] = topper_1[target]
topper_df['topper_2'] = topper_2[target]
topper_df['topper_3'] = topper_3[target]
topper_df['topper_4'] = topper_4[target]
topper_df['topper_5'] = topper_5[target]

topper_all=topper_1.copy()
topper_1 = minmax(topper_1,target)
topper_2 = minmax(topper_2,target)
topper_3 = minmax(topper_3,target)
topper_4 = minmax(topper_4,target)
topper_5 = minmax(topper_5,target)

weighed_topper=[3.0, 2.0, 1.125, 0.125, 0.0375, 3.5]
weighed_sum = np.sum(weighed_topper)

if GEOMEAN == 'Y':
    print("Geometric Mean of Topper Ensemble Calculated !!")
    topper_all[target] = stats.gmean([target_1[target], target_2[target], target_3[target], target_4[target], target_5])
else:
    topper_all[target]=(weighed_topper[1]*topper_1[target] + weighed_topper[2]*topper_2[target] + weighed_topper[3]*topper_3[target]
                      + weighed_topper[4]*topper_4[target]+ weighed_topper[5]*topper_5[target])
    print("Weighted Mean of Topper Ensemble Calculated !!")
topper_all = minmax(topper_all,target)


topper_all.to_csv('final.csv',index=False)