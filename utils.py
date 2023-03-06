import numpy as np
from scipy import sparse
import os
import torch


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_dir(path):
    try :
        print('creating ' + path)
        os.mkdir(path)
    except :
        print('path already exists')

def filter_users_and_items(sparse_mx, min_inter=20, min_inter_items = 100):
    
    N, P = sparse_mx.get_shape()
    n_interactions_per_item = sparse_mx.sum(axis = 0)
    mask_items = n_interactions_per_item > min_inter_items
    print("we keep %.2f of the items"%(mask_items.sum()/P))
    mx = sparse_mx[:, np.arange(P)[mask_items.A.ravel()]]
    
    n_interactions_per_user = mx.sum(axis = 1)
    mask = n_interactions_per_user >= min_inter
    print("we keep %.2f of the users"%(mask.sum()/N))
    
    return mx[np.arange(N)[mask.A.ravel()]]


def split_exact(cf_matrix, N_sub):
    _, P = cf_matrix.get_shape()
    matrix_context = sparse.lil_matrix((N_sub, P), dtype=np.float32)
    matrix_reward = sparse.lil_matrix((N_sub, P), dtype=np.float32)
    for i in range(N_sub):
        x, y = cf_matrix[i].nonzero()
        np.random.shuffle(y)
        n = int(len(y)/2)
        y1, y2 = y[:n], y[n:]
        matrix_context[i, y1] = 1.
        matrix_reward[i, y2] = 1.
    return matrix_context.tocsr(), matrix_reward.tocsr()


def save_tensor_spmx_dataset(tensor, sparse_mtrix, path, split = 'train'):
    save_dir = os.path.join(path, split)
    create_dir(save_dir)
    torch.save(tensor, os.path.join(save_dir, 'contexts.pt'))
    sparse.save_npz(os.path.join(save_dir, 'rewards'), sparse_mtrix, compressed=True)
    print('Saving Done into the following folder: ', save_dir)
    

def load_tensor_spmx_dataset(path, split = 'train'):
    load_dir = os.path.join(path, split)
    contexts = torch.load(os.path.join(load_dir, 'contexts.pt'))
    rewards = sparse.load_npz(os.path.join(load_dir, 'rewards.npz'))
    return contexts, rewards
    
    
def bpr_negative_sampling(rewards, n_negative = 1000):
    
    n, P = rewards.shape
    batch_potential_negatives = np.random.randint(0, P, size = [n, n_negative + 100])
    positives_to_return = np.zeros([n, 1],dtype=int)
    negatives_to_return = np.zeros([n, n_negative],dtype=int)

    for i in range(n):
        positive_actions = rewards[i].indices
        sampled_positive_a = np.random.choice(positive_actions, size = 1)

        potential_negatives = batch_potential_negatives[i]
        mask = ~np.isin(potential_negatives, positive_actions)
        sampled_negative_as = potential_negatives[mask][:n_negative]

        positives_to_return[i] = sampled_positive_a
        negatives_to_return[i] = sampled_negative_as

    return positives_to_return, negatives_to_return