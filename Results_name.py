import numpy as np
import matplotlib.pyplot as plt
import itertools


def logistic_accuracy(y,y_pred):
    accuracy0=np.mean(y[y==0]==y_pred[y==0])
    accuracy1=np.mean(y[y!=0]==y_pred[y!=0])
    return np.mean([accuracy0,accuracy1])


Names=np.load('Mice_names.npy')
names=np.load('Mice_names_data_set.npy')
 
accuracy_names=[]
accuracy_cross_val_names=[]

Folders=['R_pf','R_pupil_1','R_pupil_2']

for f,folder in enumerate(Folders):
    accuracy_names.append([])
    accuracy_cross_val_names.append([])
    for i,name in enumerate(Names):  
        accuracy_cross_val_names[f].append(np.load(folder+'/Accuracy_val_'+name+'.npy'))
        y=np.load('Trial_outcome_'+name+'.npy')
        y=np.maximum(y,0)
        y_pred=np.load(folder+'/Trial_outcome_pred_'+name+'.npy')
        accuracy_names[f].append(logistic_accuracy(y,y_pred))
    
accuracy_cross_val_names=np.array(accuracy_cross_val_names)
accuracy_names=np.array(accuracy_names)


accuracy_sets=[]
accuracy_cross_val_sets=[]

Folders=['R1','R2','R3']

for f,folder in enumerate(Folders):
    accuracy_sets.append([])
    accuracy_cross_val_sets.append([])
    for i in range(39):  
        accuracy_cross_val_sets[f].append(np.load(folder+'/Accuracy_val_{}.npy'.format(i)))
        y=np.load('Trial_outcome_{}.npy'.format(i))
        y=np.maximum(y,0)
        y_pred=np.load(folder+'/Trial_outcome_pred_{}.npy'.format(i))
        accuracy_sets[f].append(logistic_accuracy(y,y_pred))
    
accuracy_cross_val_sets=np.array(accuracy_cross_val_sets)
accuracy_sets=np.array(accuracy_sets)



#############################################################################
##                                                                         ##
##                                Analysis                                 ##
##                                                                         ##
#############################################################################



A_s=np.max(accuracy_cross_val_sets,0)
A_n=np.max(accuracy_cross_val_names,0)




#############################################################################
A_ns=[]
A_ns_mean=[]
A_ns_min=[]
A_ns_max=[]
for name in Names:
    sets=np.where(names==name)[0]
    A_ns.append(A_s[sets])
    A_ns_mean.append(np.mean(A_s[sets]))
    A_ns_min.append(np.min(A_s[sets]))
    A_ns_max.append(np.max(A_s[sets]))
A_ns_mean=np.array(A_ns_mean)
A_ns_min=np.array(A_ns_min)
A_ns_max=np.array(A_ns_max)
    
plt.figure(figsize=(12,6))
plt.scatter(Names,A_n,marker='o',c='b',label='Based on behaviour')
plt.vlines(Names, A_ns_min, A_ns_max, 'r', alpha=0.5)
plt.scatter(Names,A_ns_min,marker='_',c='r')
plt.scatter(Names,A_ns_max,marker='_',c='r')
plt.scatter(Names,A_ns_mean,marker='o',c='r',label='Based on neurons') 
plt.legend()
plt.ylabel('Accuracy')
plt.ylim([0,1]) 
plt.tight_layout()
plt.show()

#############################################################################

regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
region_colors = ['blue', 'red', 'green', 'darkblue', 'violet', 'lightblue', 'orange', 'gray']
brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP","TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                ]
B=list(itertools.chain.from_iterable(brain_groups))
Ac_color=[]
Ac_color.append('k')
for i in range(len(brain_groups)):
    for j in range(len(brain_groups[i])):
        Ac_color.append(region_colors[i])
B=np.array(['root']+B)

brain_regions=np.load('Brain_regions.npy',allow_pickle=True)
Ac=[]
for i in range(len(B)):
    Ac.append([])
        

for i in range(39):
    br=brain_regions[i,0]
    for j in range(len(br)):
        try:
            Ac[np.where(B==[(br[j].astype('U'))])[0][0]].append(A_s[i])
        except IndexError:
            print('Error')

    



Ac_mean=[]
Ac_min=[]
Ac_max=[]
for i in range(len(Ac)):
    Ac_mean.append(np.mean(Ac[i]))
    Ac_min.append(np.min(Ac[i]))
    Ac_max.append(np.max(Ac[i]))
    
    
plt.figure(figsize=(5,10))
plt.hlines(B, Ac_min, Ac_max, color=Ac_color, alpha=0.5)
plt.scatter(Ac_min,B,marker='|',c=Ac_color)
plt.scatter(Ac_max,B,marker='|',c=Ac_color)
plt.scatter(Ac_mean,B,marker='o',c=Ac_color) 
plt.xlabel('Accuracy')
plt.xlim([0,1]) 
plt.rc('ytick', labelsize=8) 
plt.tight_layout()
plt.grid()
plt.show() 
        

