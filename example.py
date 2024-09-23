import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from pyCode.PWD_DNN import pwdDNN_test, getQuantile_2side

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

alpha = 0.05
kseq = torch.tensor([1, 5, 10])
nodeSeq = torch.tensor([50, 150, 250])
rhoseq = torch.tensor([0.01, 0.1, 1])
klen = kseq.size(0)
rholen = rhoseq.size(0)
hyper_params = {'learn_rate': 0.001, 'nepoch': 1000, \
        'in_features': 1, 'hidden_layers': 3, 'node': 200, 'nodeSeq': nodeSeq}
each_projDim = False

n1 = 250
n2 = 250
sample_dim = 500
rand_generator = MultivariateNormal(torch.zeros(sample_dim, device=device), torch.eye(sample_dim, device=device))
data1 = rand_generator.sample([n1])
signal = torch.ones(1,device=device)*0.8
mean =  signal / torch.arange(1,sample_dim+1,device=device)**3
rand_generator = MultivariateNormal(mean, torch.eye(sample_dim, device=device))
data2 = rand_generator.sample([n2])

Tn = pwdDNN_test(data1, data2, kseq, rhoseq, hyper_params, each_projDim=each_projDim)
Tn = Tn.numpy()
print('test statistic: {}'.format(Tn))
if each_projDim:
    test_thresh = np.concatenate((getQuantile_2side(1-alpha, rholen).repeat(klen), getQuantile_2side(1-alpha, rholen*klen)), axis=None)
else:
    test_thresh = getQuantile_2side(1-alpha, rholen*klen)
decision = (Tn > test_thresh)    
print('decision {}'.format(decision))

    