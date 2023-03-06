import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm

twitch_dataset = pd.read_csv("full_a.csv", usecols = [0, 2], names=['user', 'streamer'])

unique_streamers = twitch_dataset['streamer'].unique()
unique_users = twitch_dataset['user'].unique()
dict_streamers = {streamer:i for i,streamer in enumerate(unique_streamers)}

N = len(unique_users)
P = len(unique_streamers)

print("number of users: ", N)
print("number of items: ", P)

cf_matrix = sparse.lil_matrix((N, P), dtype=np.float32)

values = twitch_dataset.values

for line in tqdm(values):
    cf_matrix[line[0] - 1, dict_streamers[line[1]]] += 1.
    
cf_matrix = cf_matrix.tocsr()

sparse.save_npz('../datasets/twitch_sparse.npz', cf_matrix, compressed=True)