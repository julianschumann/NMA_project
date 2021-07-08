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
stim_time=0.5
for i,dat in enumerate(alldat):
    number_trials[i]=len(dat['contrast_left'])
    number_neurons[i]=len(dat['brain_area'])
    neurons.append(dat['brain_area'])
    trial_outcome.append(dat['feedback_type'])
    spike_data.append(dat['spks'])
    response_time.append(dat['response_time'])
number_trials=number_trials.astype('int')
number_neurons=number_neurons.astype('int')

    
