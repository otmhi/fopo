import numpy as np
from scipy import sparse
from tqdm import tqdm
import os

import scipy.sparse.linalg as sp_l
import faiss
import torch 

from utils import filter_users_and_items, split_exact, save_tensor_spmx_dataset, create_dir

# Manage all hyperparameters :
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--dataset", type=str, default='twitch')
parser.add_argument("--N_sub", type=int, default=500_000)
parser.add_argument("--K", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)

args = vars(parser.parse_args())
print(args)

seed = args['seed']
np.random.seed(seed)

data_path = 'datasets/' + args['dataset'] + '.npz'

cf_sparse = sparse.load_npz(data_path)
cf_sparse_implicit = 1.* (cf_sparse>0)
N, P = cf_sparse.get_shape()

min_inter_users, min_inter_items = 20, 0
print('we only select users with at least %d interactions'%min_inter_users)
print('we only select items with at least %d interactions'%min_inter_items)

cf_matrix = filter_users_and_items(cf_sparse_implicit, min_inter_users, min_inter_items)

N_f, P = cf_matrix.get_shape()
print("the number of kept users is : ", N_f)
print("the number of kept items is : ", P)
print('The average user interaction with streamers is :', np.mean(np.sum(cf_matrix > 0, axis = 1)))


N_sub = min(args['N_sub'], N_f)
print("we subsample the dataset to only retain %d users"%N_sub)
matrix_context, matrix_reward = split_exact(cf_matrix, N_sub)


split = 0.8
val_split = 0.1

print("we take a proportion of %.2f for the training users"%split)
idx = np.arange(N_sub)

train_prop = int(split * N_sub)
val_prop = int(val_split * N_sub)

N_train = train_prop
N_test = val_prop
print('The training size :', N_train)
print('The testing size :', N_test)

np.random.shuffle(idx)
idx_train, idx_val, idx_test = idx[:train_prop], idx[train_prop:train_prop + val_prop], idx[train_prop + val_prop:]


context_train, reward_train = matrix_context[idx_train], matrix_reward[idx_train]
context_val, reward_val = matrix_context[idx_val], matrix_reward[idx_val]
context_test, reward_test = matrix_context[idx_test], matrix_reward[idx_test]


train_interactions = context_train + reward_train
K = args['K']

dataset_path = os.path.join('experiments', args['dataset'])
create_dir(dataset_path)

exp_path = os.path.join(dataset_path, 'N:%d_K:%d_seed:%d'%(N_sub, K, seed))
create_dir(exp_path)

exp_data_path = os.path.join(exp_path, 'data')
create_dir(exp_data_path)

print('we use SVD to lower the dimension to %d'%K)

U, S, V_t = sp_l.svds(train_interactions, k = K)
prod_emb = np.float32(V_t.T.copy(order='C'))
np.save(os.path.join(exp_data_path, 'prod_emb.npy'), prod_emb)

print("SVD Done and Saved")


normalizer_train = sparse.diags(1/context_train.sum(axis = 1).A.ravel())
normalizer_val = sparse.diags(1/context_val.sum(axis = 1).A.ravel())
normalizer_test = sparse.diags(1/context_test.sum(axis = 1).A.ravel())

context_emb_train = normalizer_train @ context_train @ prod_emb
context_emb_val = normalizer_val @ context_val @ prod_emb
context_emb_test = normalizer_test @ context_test @ prod_emb

train_c, val_c, test_c = torch.tensor(context_emb_train), torch.tensor(context_emb_val), torch.tensor(context_emb_test)


save_tensor_spmx_dataset(train_c, reward_train, exp_data_path, split = 'train')
save_tensor_spmx_dataset(val_c, reward_val, exp_data_path, split = 'val')
save_tensor_spmx_dataset(test_c, reward_test, exp_data_path, split = 'test')


print('Creating the index')
index = faiss.IndexHNSWFlat(K, 32, faiss.METRIC_INNER_PRODUCT)
index.add(prod_emb)

faiss.write_index(index, os.path.join(exp_data_path, 'index'))
print('Creating and Saving Done')