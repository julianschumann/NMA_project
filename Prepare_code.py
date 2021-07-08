import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def simple_data(spike, response_time, stim_time):
    (n_neurons, n_samples, n_time_bin)=spike.shape 
    stim_time_index=int(100*stim_time)
    response_time_index=(np.maximum(response_time[:,0],stim_time)*100).astype('int')+1
    mean_spike_count=np.zeros((n_samples,n_neurons))
    for i in range(n_samples):
        for j in range(n_neurons):
            mean_spike_count[i,j]=np.mean(spike[j,i,stim_time_index:response_time_index[i]])
    return mean_spike_count

def complex_data(spike, response_time, stim_time,n_stim, n_resp):
    (n_neurons, n_samples, n_time_bin)=spike.shape 
    stim_time_index=int(100*stim_time)
    response_time_index=(np.maximum(response_time[:,0],stim_time)*100).astype('int')+1
    mean_spike_count=np.zeros((n_samples,n_neurons,n_stim+n_resp))
    for i in range(n_samples):
        for j in range(n_neurons):
            viable=spike[j,i,stim_time_index:response_time_index[i]]
            mean_spike_count[i,j,:min(n_stim,len(viable))]=viable[:min(n_stim,len(viable))]
            mean_spike_count[i,j,n_stim+n_resp-min(n_resp,len(viable)):]=viable[len(viable)-min(n_stim,len(viable)):]
    return mean_spike_count



def logistic_regression(X,y):
    C=np.logspace(-5,5,11)
    accuracies=[]
    for i,c in enumerate(C):
        print('Step {} wiht c={:1.0e}'.format(i,c))
        log_reg_l1 = LogisticRegression(penalty="l1", C=c, solver="saga", max_iter=5000)
        accuracies.append(np.mean(cross_val_score(log_reg_l1, X, y, cv=5)))
    c_opt=C[np.argmax(accuracies)]
    log_reg_l1 = LogisticRegression(penalty="l1", C=c_opt, solver="saga", max_iter=5000).fit(X,y)
    return log_reg_l1,c_opt




data_set=16 # 39 runs, so data_set in {0,...,38}
n_stim=100 
n_resp=100

Type=False # True: complex, False: Mean

spikes=np.load('Spike_data_{}.npy'.format(data_set))
t_r=np.load('Response_time_{}.npy'.format(data_set))
y=np.load('Trial_outcome_{}.npy'.format(data_set))
t_s=np.load('Stimulation_time.npy')
y=np.maximum(y,0)

if Type:
    X=complex_data(spikes, t_r, t_s,n_stim, n_resp)
    
    X=X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
else:
    X=simple_data(spikes,t_r,t_s)

Model,c_opt=logistic_regression(X,y)

y_pred=Model.predict(X)
accuracy=np.mean(y==y_pred)

plt.figure()
plt.scatter(np.arange(len(y)),y,marker='o',label='original')
plt.scatter(np.arange(len(y)),y_pred,marker='x',label='prediction')
plt.title('Data set {}, accuracy={:1.2f}'.format(data_set,accuracy))
plt.xlabel('trial')
plt.legend()
plt.tight_layout() 
plt.show()
