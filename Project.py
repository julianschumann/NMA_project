import os, requests
from matplotlib import rcParams 
from matplotlib import pyplot as plt
import numpy as np

rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] =15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True


fname = []
for j in range(3):
    fname.append('steinmetz_part{}.npz'.format(j))
url = ["https://osf.io/agvxh/download"]
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")

for j in range(len(url)):
    if not os.path.isfile(fname[j]):
        try:
            r = requests.get(url[j])
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname[j], "wb") as fid:
                    fid.write(r.content)
          
alldat = np.array([])
for j in range(len(fname)):
    alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))
  

###########################################################################################


number_trials=np.zeros(len(alldat))
trial_outcome=[]
number_neurons=np.zeros(len(alldat))
neurons=[]
spike_data=[]
response_time=[]
pupil_area=[]
names=[]
face_energy=[]
brain_regions=[]
stim_time=0.5
for i,dat in enumerate(alldat):
    number_trials[i]=len(dat['contrast_left'])
    number_neurons[i]=len(dat['brain_area'])
    neurons.append(dat['brain_area'])
    trial_outcome.append(dat['feedback_type'])
    spike_data.append(dat['spks'])
    response_time.append(dat['response_time'])
    pupil_area.append(dat['pupil'][0])
    names.append(dat['mouse_name'])
    face_energy.append(dat['face'][0])
    brain_regions.append(np.unique(dat['brain_area'],return_counts=True))
number_trials=number_trials.astype('int')
number_neurons=number_neurons.astype('int')
names=np.array(names)
brain_regions=np.array(brain_regions)
np.save('Brain_regions.npy',brain_regions)
Names=np.unique(names)

for i in range(len(alldat)):
    np.save('Spike_data_{}.npy'.format(i),spike_data[i])
    np.save('Response_time_{}.npy'.format(i),response_time[i])
    np.save('Trial_outcome_{}.npy'.format(i),trial_outcome[i])
np.save('Stimulation_time.npy',stim_time)

    

pupil_area_names=[]
face_energy_names=[]
trial_outcome_names=[]
response_time_names=[]
for i,name in enumerate(Names):
    name_id=np.where(names==name)[0]
    pupili=[]
    triali=[]
    facei=[]
    timei=[]
    braini=[]
    for j in name_id:
        pupili.append(pupil_area[j])
        triali.append(trial_outcome[j])
        facei.append(face_energy[j])
        timei.append(response_time[j])
    pupil_area_names.append(np.concatenate(pupili,0))
    trial_outcome_names.append(np.concatenate(triali,0))
    face_energy_names.append(np.concatenate(facei,0))
    response_time_names.append(np.concatenate(timei,0))
    
    np.save('Pupil_area_'+name+'.npy',pupil_area_names[i])
    np.save('Trial_outcome_'+name+'.npy',trial_outcome_names[i])
    np.save('Face_energy_'+name+'.npy',face_energy_names[i])
    np.save('Response_time'+name+'.npy',response_time_names[i])
np.save('Mice_names.npy',np.array(Names))
np.save('Mice_names_data_set.npy',np.array(names))
