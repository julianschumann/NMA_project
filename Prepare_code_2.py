import numpy as np
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



def logistic_accuracy(y,y_pred):
    accuracy0=np.mean(y[y==0]==y_pred[y==0])
    accuracy1=np.mean(y[y!=0]==y_pred[y!=0])
    return np.mean([accuracy0,accuracy1])

    
def logistic_gradient(X,y,beta,theta):
    X0=X[y==0]
    X1=X[y!=0]
    
    Z0=np.sum(X0*theta[np.newaxis,1:],1)+theta[0]
    Sigma0=1/(1+np.exp(-Z0))
    L0=(1-Sigma0)
    LL0=-np.log(L0+1e-10)
    LL0=0.5*len(y)*np.mean(LL0)
    
    Z1=np.sum(X1*theta[np.newaxis,1:],1)+theta[0]
    Sigma1=1/(1+np.exp(-Z1))
    L1=Sigma1
    LL1=-np.log(L1+1e-10)
    LL1=0.5*len(y)*np.mean(LL1)
    
    
    LLB=beta*np.sum(np.abs(theta)+0.5*theta**2)
    loss=LLB+LL0+LL1
    
    dLL0dL0=-0.5*len(y)/np.sum(y==0)/(L0+1e-10)
    dL0dSigma0=-1
    dSigma0dZ0=np.exp(-Z0)/(1+np.exp(-Z0))**2
    dLL0dZ0=dLL0dL0*dL0dSigma0*dSigma0dZ0 #
    dZ0dtheta=np.concatenate((np.ones((np.sum(y==0),1)),X0),1)
    dLL0dtheta0=np.sum(dZ0dtheta*dLL0dZ0[:,np.newaxis],0)
    
    dLL1dL1=-0.5*len(y)/np.sum(y!=0)/(L1+1e-10)
    dL1dSigma1=1
    dSigma1dZ1=np.exp(-Z1)/(1+np.exp(-Z1))**2
    dLL1dZ1=dLL1dL1*dL1dSigma1*dSigma1dZ1 #
    dZ1dtheta=np.concatenate((np.ones((np.sum(y!=0),1)),X1),1)
    dLL1dtheta1=np.sum(dZ1dtheta*dLL1dZ1[:,np.newaxis],0)
    
    dLLBdtheta=beta*(np.sign(theta)+theta)
    
    G=dLLBdtheta+dLL0dtheta0+dLL1dtheta1
    return G,loss  
    
    
def neg_log_likelyhood_train(X,y,beta,iterations):
    theta=np.zeros(X.shape[1]+1)
    # X= n times d
    # y= n
    # beta=scalar
    # theta= d+1
    eta=1
    betam=0.5
    betav=0.75 
    betamh=0.5 
    betavh=0.75 
    m=np.zeros_like(theta)
    v=np.zeros_like(theta)
    for i in range(iterations):
        G,_=logistic_gradient(X,y,beta,theta)
        m=betam*m+(1-betam)*G
        v=betav*v+(1-betav)*G*G
        mh=m/(1-betamh)
        vh=v/(1-betavh)
        diff=eta*mh/(np.sqrt(vh)+1e-6)
        if np.max(np.abs(diff))<1e-6:
            break
        theta=theta-diff
    return theta
    
    
def logistic_predictor(X,theta):
    #train==true: continuous output, train==False, discrete output 
    z=np.sum(X*theta[np.newaxis,1:],1)+theta[0]
    sigma=1/(1+np.exp(-z))
    return np.floor(sigma+0.5).astype('int')


def cross_validate(X,y,beta,k):
    n=len(y)
    nk=int(n/k)
    Index=np.arange(n)
    np.random.shuffle(Index)
    accuracies=np.zeros(k)
    for i in range(k):
        Index_val=Index[i*nk:(i+1)*nk]
        Index_train=np.concatenate((Index[:i*nk],Index[(i+1)*nk:]),0)
        X_train=X[Index_train,:]
        y_train=y[Index_train]
        X_val=X[Index_val,:]
        y_val=y[Index_val]
        theta=neg_log_likelyhood_train(X_train,y_train,beta,1000)
        y_pred=logistic_predictor(X_val,theta)
        accuracies[i]=logistic_accuracy(y_val,y_pred)
    return np.mean(accuracies)
    


def logistic_regression(X,y):
    Beta=np.logspace(-5,5,11)
    accuracies=[]
    for i,beta in enumerate(Beta):
        print('Step {} wiht beta={:1.0e}'.format(i,beta))
        accuracies.append(cross_validate(X,y,beta,5))
    beta_opt=Beta[np.argmax(accuracies)]
    theta=neg_log_likelyhood_train(X,y,beta_opt,2500)
    y_pred=logistic_predictor(X,theta)
    return y_pred,beta_opt,np.max(accuracies),theta


make_data=False

if make_data:

    for data_set in range(39):
        
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
        
        
        y_pred,beta,accuracy,theta=logistic_regression(X,y)
        
        np.save('R2/Trial_outcome_pred_{}.npy'.format(data_set),y_pred)
        np.save('R2/Accuracy_val_{}.npy'.format(data_set),accuracy)
        np.save('R2/C_{}.npy'.format(data_set),beta)
        np.save('R2/Theta_{}.npy'.format(data_set),beta)
        plt.figure()
        plt.scatter(np.arange(len(y)),y,marker='o',label='original')
        plt.scatter(np.arange(len(y)),y_pred,marker='x',label='prediction')
        plt.title('Data set {}, accuracy={:1.2f}, cross val accuracy={:1.2f} \n l1-regularization with beta={:1.1e}'.format(data_set,logistic_accuracy(y,y_pred),accuracy,beta))
        plt.xlabel('trial')
        plt.legend()
        plt.tight_layout() 
        plt.show()
else:  
    accuracy=[]
    Beta=[]
    accuracy_cross_val=[]
    for data_set in range(39):   
        Beta.append(np.load('R2/C_{}.npy'.format(data_set)))
        y=np.load('Trial_outcome_{}.npy'.format(data_set))
        accuracy_cross_val.append(np.load('R2/Accuracy_val_{}.npy'.format(data_set)))
        y=np.maximum(y,0)
        y_pred=np.load('R2/Trial_outcome_pred_{}.npy'.format(data_set))
        accuracy.append(logistic_accuracy(y,y_pred))
        plt.figure(figsize=(6,3))
        plt.scatter(np.arange(len(y)),y,marker='o',label='original')
        plt.scatter(np.arange(len(y)),y_pred,marker='x',label='prediction')
        plt.title('Data set {}, accuracy={:1.2f}, cross val accuracy={:1.2f} \n l1-regularization with beta={:1.1e}'.format(data_set,accuracy[data_set],accuracy_cross_val[data_set],Beta[data_set]))
        plt.xlabel('trial')
        plt.legend()
        plt.tight_layout() 
        plt.show()
    accuracy_cross_val=np.array(accuracy_cross_val)
    accuracy=np.array(accuracy)
    Beta=np.array(Beta)
    
    
