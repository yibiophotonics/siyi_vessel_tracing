# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:35:50 2024

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from post_processing.post import Coloring

#%%
path = "C:/Users/Administrator/OneDrive - Johns Hopkins/2023 Siyi Chen vessel tracing/results/patches/InSegNN/"
results = pd.read_excel("".join((path,"statistics result of patches.xlsx")))
#%
sbd = results["SBD"]
sensitivity = results["Dice_true"]
specificity = results["Dice_pred"]
sample_size = np.sqrt(results.shape[0])
#%
x = ["SBD","Sensitivity","Specificity"]
y = [sbd.mean(),sensitivity.mean(),specificity.mean()]
err = [sbd.std()/sample_size,sensitivity.std()/sample_size,specificity.std()/sample_size]
#%
fig,ax = plt.subplots(figsize=(5,5))
plt.bar(x,y,yerr=err,capsize=10)
plt.ylim((0.65,0.85))
plt.ylabel("Segmentation results",weight="bold")
plt.title("InSegNN with temporal module on new dataset",y=1.1,weight="bold")
ax.spines[['right', 'top']].set_visible(False)
for i,v in enumerate(y):
    plt.text(i, v+err[i]+0.01, '%.3f' % v, ha="center",weight="bold")
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
plt.tight_layout()
plt.savefig(path + "results2.png",dpi=500)

#%%
path = "D:/siyi_vessle/ji_amir_resize/whole_image/vessel_match/prediction/10/"
vessels = os.listdir(path)
#%
vessel = io.imread("".join((path,vessels[0])))
mask = np.empty((len(vessels),vessel.shape[0],vessel.shape[1]))
for i in vessels:
    vessel = io.imread("".join((path,i)))
    if vessel.ndim > 2:
        vessel = vessel[:,:,0]
    vessel = vessel/vessel.max()
    if vessel.sum() > vessel.size/2:
        vessel = 1 - vessel
    idx = int(i[0])
    mask[idx] = vessel
    
#%
ins_color_img, colors = Coloring(mask,plt.cm.Set2,'image_channel')
#%
plt.imsave("".join((path,"all.png")),ins_color_img)

#%%
path = "D:/siyi_vessle/ji_amir_resize/whole_image/vessel_match/"
results = pd.read_excel("".join((path,"result_stats.xlsx"))) 
old_data = results[results["Subject"]>21]
new_data = results[results["Subject"]<30]
#%
new_precision = new_data["TP"] / (new_data["TP"] + new_data["FP"])
new_sensitivity = new_data["TP"] / (new_data["TP"] + new_data["FN"])
old_precision = old_data["TP"] / (old_data["TP"] + old_data["FP"])
old_sensitivity = old_data["TP"] / (old_data["TP"] + old_data["FN"])
precision = results["TP"] / (results["TP"] + results["FP"])
sensitivity = results["TP"] / (results["TP"] + results["FN"])
#%
stats = pd.DataFrame()
stats["label"] = ['Original dataset','New dataset','Total']
stats["Precision"] = [old_precision.mean(),new_precision.mean(),precision.mean()]
stats["precision_std"] = [old_precision.std()/np.sqrt(old_precision.size),new_precision.std()/np.sqrt(new_precision.size),precision.std()/np.sqrt(precision.size)]
stats["Sensitivity"] = [old_sensitivity.mean(),new_sensitivity.mean(),sensitivity.mean()]
stats["sensitivity_std"] = [old_sensitivity.std()/np.sqrt(old_sensitivity.size),new_sensitivity.std()/np.sqrt(new_sensitivity.size),sensitivity.std()/np.sqrt(sensitivity.size)]
#%
stats.plot.bar(x='label',y=['Precision','Sensitivity'],yerr=stats[['precision_std','sensitivity_std']].T.values,capsize=5)
ax = plt.gca()
ax.set_xlabel("")
ax.tick_params(axis='x', labelrotation=0)
ax.set_ylim(top=1.2)
ax.set_title("Global tracing results")
for p in ax.patches:
    ax.annotate('%.3f' % p.get_height(), (p.get_x(), p.get_height() + 0.05))
plt.savefig(path + "results2.png",dpi=500)

#%%
path = "C:/Users/Administrator/OneDrive - Johns Hopkins/2023 Siyi Chen vessel tracing/whole_image/input/multiple start point/"
results = pd.read_excel("".join((path,"result_stats.xlsx"))) 
#%
prec = []
sens = []
subjects = np.unique(results["Subject"])
for subject in subjects:
    result_s = results[results["Subject"]==subject]
    vessels = np.unique(result_s["Vessel"])
    for vessel in vessels:
        result_v = result_s[result_s["Vessel"]==vessel]
        precision = result_v["TP"] / (result_v["TP"] + result_v["FP"])
        sensitivity = result_v["TP"] / (result_v["TP"] + result_v["FN"])
        
        #prec.append(precision.max() - precision.min())
        #sens.append(sensitivity.max() - sensitivity.min())   
        prec.append(precision.var())
        sens.append(sensitivity.var())   
#%
precision = np.array(prec)
sensitivity = np.array(sens)
#%
fig,ax = plt.subplots(figsize=(5,5))
plt.boxplot([precision,sensitivity],patch_artist=True,labels=['Precision','Sensitivity'])
plt.text(0.65, np.median(precision), '%.3f' % np.median(precision), weight="bold")
plt.text(1.65, np.median(sensitivity), '%.3f' % np.median(sensitivity), weight="bold")
plt.ylabel('Variance of result over 7 annotation points',weight="bold")
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
        
plt.savefig(path + "results.png",dpi=500)

        





















