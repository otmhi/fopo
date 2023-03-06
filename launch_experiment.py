import pandas as pd
import numpy as np
from scipy import sparse
import os

import scipy.sparse.linalg as sp_l
import faiss
import torch 
from torch import nn, optim
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Manage all hyperparameters :
from argparse import ArgumentParser
from utils import *
from algorithms import *

parser = ArgumentParser()

parser.add_argument("--exp_folder", type=str, default='experiments/twitch/N:300000_K:1000_seed:0')
parser.add_argument("--method", type=str, default='uniform')
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--trunc_at", type=int, default=256)
parser.add_argument("--eps", type=float, default=.8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=0)


args = vars(parser.parse_args())
print(args)

seed = args['seed']
np.random.seed(seed)
torch.manual_seed(seed)

print('Loading the different splits')
exp_folder_path = args['exp_folder']
data_folder = os.path.join(exp_folder_path, 'data')

contexts_train, rewards_train = load_tensor_spmx_dataset(data_folder, split = 'train')
contexts_val, rewards_val = load_tensor_spmx_dataset(data_folder, split = 'val')
contexts_test, rewards_test = load_tensor_spmx_dataset(data_folder, split = 'test')

print('Loading the embeddings')
prod_emb = np.load(os.path.join(data_folder, 'prod_emb.npy'))
P = prod_emb.shape[0]
print('number of products is %d'%P)

print('Loading the index')
index = faiss.read_index(os.path.join(data_folder, 'index'))


print('Define the learning algorithms')

epochs = 10
bsize = 32
reg = 1e-10

method = args['method']
lr = args['lr']
n_samples = args['n_samples']
eps = args['eps']
trunc_at = args['trunc_at']



policy = policy_model(prod_emb).cuda()

if method == 'exact' :
    
    path = method + '_Ns:%d_Lr:%.6f'%(n_samples, lr)
    
    print('-------------Exact Reinforce---------------')
    
    train_log, val_log_index, val_log, exec_time = exact_reinforce(policy, index, n_samples, contexts_train, rewards_train, 
                                                    epochs, bsize, lr, reg, contexts_val, rewards_val)
    
    
elif method == 'uniform' : 
    
    path = method + '_Ns:%d_Lr:%.6f'%(n_samples, lr)
    
    print('-------------Uniform SNIS---------------')
    
    train_log, val_log_index, val_log, exec_time = uniform_snips_approximate_reinforce(policy, index, P, n_samples, contexts_train, rewards_train, 
                                                                        epochs, bsize, lr, reg, contexts_val, rewards_val)
      
elif method == 'mixture':
    
    path = method + '_Ns:%d_T@:%d_Eps:%.1f_Lr:%.6f'%(n_samples, trunc_at, eps, lr)
    
    print('-------------Mixture SNIS %.1f---------------'%eps)
    
    train_log, val_log_index, val_log, exec_time = mixture_snips_approximate_reinforce(policy, index, P, n_samples, trunc_at, eps, 
                                                                        contexts_train, rewards_train, epochs, bsize, lr, reg,
                                                                        contexts_val, rewards_val)
    
else:
    print("method not supported")
    print("method should be either exact, uniform, mixture")



test_log_index, test_log = test_policy(policy, index, contexts_test, rewards_test, 512)
print('Test Results : INDEX %.4f, TRUE %.4f '%(test_log_index, test_log))
print('Execution Time : %.4f seconds'%exec_time)
result_path = os.path.join(exp_folder_path, 'results/' + path)
create_dir(result_path)

train_logs_dict = {method: train_log}
train_logs = pd.DataFrame(train_logs_dict)
train_logs.to_csv(result_path + '/train_logs.csv', index=False)


val_logs_dict = {method: val_log}
val_logs = pd.DataFrame(val_logs_dict)
val_logs.to_csv(result_path + '/val_logs.csv', index=False)

val_logs_dict = {method: val_log_index}
val_logs = pd.DataFrame(val_logs_dict)
val_logs.to_csv(result_path + '/val_logs_index.csv', index=False)

exec_time_dict = {method: [exec_time, test_log]}
exec_logs = pd.DataFrame(exec_time_dict)
exec_logs.to_csv(result_path + '/exec_test_logs.csv', index=False)
