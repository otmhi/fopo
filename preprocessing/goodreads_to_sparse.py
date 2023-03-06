import pandas as pd
from tqdm import tqdm
from scipy import sparse
import numpy as np

df = pd.read_csv('goodreads_interactions.csv')

df = df[df['is_read'] > 0.5]
users = df['user_id'].unique()
items = df['book_id'].unique()

dict_items = {item:i for i,item in enumerate(items)}
dict_users = {user:i for i,user in enumerate(users)}

N = len(users)
P = len(items)

print("number of users: ", N)
print("number of items: ", P)

cf_matrix = sparse.lil_matrix((N, P), dtype=np.float32)

values = df[['user_id', 'book_id']].values

for line in tqdm(values):
    user, item = line[0], line[1]
    cf_matrix[dict_users[user], dict_items[item]] += 1.
    
cf_matrix = cf_matrix.tocsr()

sparse.save_npz('../datasets/goodreads.npz', cf_matrix, compressed=True)