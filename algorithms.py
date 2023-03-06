import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn



def test_policy(policy, index, contexts, rewards, bsize):
    """
    Function to test the reward collected by a policy on a the validation/test set.
    """
    
    greed_reward_index, greed_reward = 0., 0.
    N_test = contexts.size(0)
    idxlist = np.arange(N_test)
    
    for bnum, st_idx in tqdm(enumerate(range(0, N_test, bsize))):
            end_idx = min(st_idx + bsize, N_test)
            indices = idxlist[st_idx:end_idx]
            len_indices = len(indices)
            help_broadcast = np.arange(len_indices)
            
            r = rewards[indices]
            c = contexts[indices].cuda()
            
            _, top_actions_index = index.search(policy.Xtransformed(c).cpu().detach().numpy(), k = 1)
            top_rewards_index = rewards[indices[:,np.newaxis], top_actions_index].A
            
            top_actions = policy.argmax(c).cpu().numpy()
            top_rewards = rewards[indices].A[help_broadcast[:,np.newaxis], top_actions]
            
            
            r_a_index = top_rewards_index.sum()
            r_a = top_rewards.sum()

            greed_reward_index += r_a_index/N_test
            greed_reward += r_a/N_test
                
    return greed_reward_index, greed_reward



class policy_model(torch.nn.Module):
    """
    Define a Policy with the MIPS structure:
    Linear user transformation with fixed embeddings.
    """
    def __init__(self, emb):
        super(policy_model, self).__init__()
        self.emb = torch.Tensor(emb.T).cuda()
        self.K, _ = self.emb.shape
        self.theta = torch.nn.Parameter(0.005 * torch.randn(self.K, self.K))
        self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_transformed = torch.matmul(x, self.theta)
        log_unnormalized = torch.matmul(x_transformed, self.emb)
        return nn.functional.softmax(log_unnormalized, dim = -1), log_unnormalized
    
    def sample(self, x, n_samples = 1):
        x_transformed = torch.matmul(x, self.theta)
        log_unnormalized = torch.matmul(x_transformed, self.emb)
        scores = torch.exp(log_unnormalized - log_unnormalized.max())
        actions = torch.multinomial(scores, n_samples, replacement=True)
        return actions
    
    def argmax(self, x):
        x_transformed = torch.matmul(x, self.theta)
        log_unnormalized = torch.matmul(x_transformed, self.emb)
        return torch.argmax(log_unnormalized, dim = 1, keepdim = True)
    
    def Xtransformed(self, x):
        x_transformed = torch.matmul(x, self.theta)
        return x_transformed
    


def exact_reinforce(pi, index, n_samples, contexts, rewards, epochs, bsize, lr, reg,
                    val_contexts, val_rewards, baseline = False):
    """
    Training the policy with a naive REINFORCE algorithm.
    Scales linearly in the catalog size.
    The index is only used for evaluating (fast) the policy.
    """
    
    optimizer = torch.optim.Adam(pi.parameters(), lr=lr, weight_decay=reg)
    N_train = contexts.size(0)
    idxlist = np.arange(N_train)
    
    train_m_rewards = []
    val_max, val_max_index = [0.]*epochs, [0.]*epochs
    
    true_duration = 0.
    
    for i in range(epochs):
        print("epoch number %d"%i)
        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in tqdm(enumerate(range(0, N_train, bsize))):
            base_time = time.time()
            
            end_idx = min(st_idx + bsize, N_train)
            indices = idxlist[st_idx:end_idx]
            indices_len = len(indices)
            
            X = contexts[indices].cuda()
            probs, _ = pi(X)
            with torch.no_grad():
                actions = torch.multinomial(probs, n_samples, replacement=True)
                r_a = torch.Tensor(rewards[indices[:,np.newaxis], actions.cpu()].A).cuda()
                if baseline : r_a = r_a - torch.mean(r_a, dim=-1, keepdim=True)
            
            help_broadcast = np.arange(indices_len)[:,np.newaxis]
            loss = -torch.mean(r_a * torch.log(probs[help_broadcast, actions]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            true_duration += time.time() - base_time
            
            _, top_actions = index.search(pi.Xtransformed(X).cpu().detach().numpy(), k = 1)
            top_rewards = rewards[indices[:,np.newaxis], top_actions].A
            r_a = top_rewards.mean()
            train_m_rewards.append(r_a)
            
        print('Computing metrics on val')
        max_perf_index, max_perf = test_policy(pi, index, val_contexts, val_rewards, 512)
        val_max_index[i], val_max[i] = max_perf_index, max_perf
        print('Argmax reward on the validation : INDEX %.4f, TRUE %.4f'%(max_perf_index, max_perf))
        
    return train_m_rewards, val_max_index, val_max, true_duration


def uniform_snips_approximate_reinforce(pi, index, P, n_samples, contexts, rewards, epochs, bsize, lr, reg,
                                        val_contexts, val_rewards):
    """
    Training the policy with a the new covariance gradient, with a uniform proposal (epsilon equal to 0)
    Training speed does not depend on the size of the catalog.
    The index is only used for evaluating (fast) the policy.
    """
    
    optimizer = torch.optim.Adam(pi.parameters(), lr=lr, weight_decay=reg)
    N_train = contexts.size(0)
    idxlist = np.arange(N_train)

    val_max, val_max_index = [0.]*epochs, [0.]*epochs
    train_m_rewards = []
    
    true_duration = 0.
    
    for i in range(epochs):
        print("epoch number %d"%i)
        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in tqdm(enumerate(range(0, N_train, bsize))):
            
            base_time = time.time()
            
            end_idx = min(st_idx + bsize, N_train)
            indices = idxlist[st_idx:end_idx]
            len_indices = len(indices)
            
            X = contexts[indices].cuda()
            x_transformed = torch.matmul(X, pi.theta)
            a_samples = torch.randint(P, [len_indices, n_samples])
            a_embs = pi.emb[:, a_samples]
            
            log_p_tilde = torch.einsum('ij,jik->ik', x_transformed, a_embs)
            with torch.no_grad():
                ws = torch.nn.functional.softmax(log_p_tilde, dim=-1) # SNIPS
            
            r_a = torch.Tensor(rewards[indices[:,np.newaxis], a_samples].A).cuda()

            mean_log_p_tilde, mean_rewards = torch.sum(ws * log_p_tilde, dim=-1, keepdim=True), torch.sum(ws * r_a, dim=-1, keepdim=True)
            loss = - torch.mean(torch.sum(ws * (log_p_tilde - mean_log_p_tilde) * (r_a - mean_rewards), dim=-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            true_duration += time.time() - base_time
            
            _, top_actions = index.search(pi.Xtransformed(X).cpu().detach().numpy(), k = 1)
            top_rewards = rewards[indices[:,np.newaxis], top_actions].A
            r_a = top_rewards.mean()
            train_m_rewards.append(r_a)
        
        print('Computing metrics on val')
        max_perf_index, max_perf = test_policy(pi, index, val_contexts, val_rewards, 512)
        val_max_index[i], val_max[i] = max_perf_index, max_perf
        print('Argmax reward on the validation : INDEX %.4f, TRUE %.4f'%(max_perf_index, max_perf))
        
    return train_m_rewards, val_max_index, val_max, true_duration


def mixture_snips_approximate_reinforce(pi, index, P, n_samples, trunc_at, eps, 
                                        contexts, rewards, epochs, bsize, lr, reg,
                                        val_contexts, val_rewards):
    """
    Training the policy with a the new covariance gradient alongside a proposal based on MIPS
    Training speed depends logarithmically on the catalog size
    The index is used for training and evaluating the policy.
    """
    
    optimizer = torch.optim.Adam(pi.parameters(), lr=lr, weight_decay=reg)
    N_train = contexts.size(0)
    idxlist = np.arange(N_train)
    proposal_probs = torch.ones(bsize, P).cuda() * eps/P
    
    val_max, val_max_index = [0.]*epochs, [0.]*epochs
    train_m_rewards = []
    
    true_duration = 0.
    
    for i in range(epochs):
        print("epoch number %d"%i)
        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in tqdm(enumerate(range(0, N_train, bsize))):
            
            base_time = time.time()
            
            end_idx = min(st_idx + bsize, N_train)
            indices = idxlist[st_idx:end_idx]
            len_indices = len(indices)
            help_broadcast = np.arange(len_indices)[:,np.newaxis]
            
            X = contexts[indices].cuda()
            x_transformed = torch.matmul(X, pi.theta)

            query = x_transformed.cpu().detach().numpy()
            topK_scores, topK_indices = index.search(query, k = trunc_at)


            topK_indices, topK_scores = torch.tensor(topK_indices).cuda(), torch.tensor(topK_scores).cuda()
            topK_probs = torch.nn.functional.softmax(topK_scores, dim=-1)

            proposal_probs[help_broadcast, topK_indices] += (1. - eps) * topK_probs

            uni_n_samples = int(n_samples * eps)
            a_samples_uni = torch.randint(P, [len_indices, uni_n_samples]).cuda()
            a_samples_topK = topK_indices[help_broadcast, torch.multinomial(topK_probs, n_samples - uni_n_samples, replacement=True)]
            
            a_samples = torch.cat([a_samples_uni, a_samples_topK], dim=1)

            a_embs = pi.emb[:, a_samples]
            log_p_tilde = torch.einsum('ij,jik->ik', x_transformed, a_embs)

            # SNIPS
            with torch.no_grad():
                log_ws_tilde = log_p_tilde - torch.log(proposal_probs[help_broadcast, a_samples])
                ws = torch.nn.functional.softmax(log_ws_tilde, dim=-1)
            
            r_a = torch.Tensor(rewards[indices[:,np.newaxis], a_samples.cpu()].A).cuda()
            mean_log_p_tilde, mean_rewards = torch.sum(ws * log_p_tilde, dim=-1, keepdim=True), torch.sum(ws * r_a, dim=-1, keepdim=True)
            loss = - torch.mean(torch.sum(ws * (log_p_tilde - mean_log_p_tilde) * (r_a - mean_rewards), dim=-1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            proposal_probs[help_broadcast, topK_indices] = eps/P
            
            true_duration += time.time() - base_time
            
            _, top_actions = index.search(pi.Xtransformed(X).cpu().detach().numpy(), k = 1)
            top_rewards = rewards[indices[:,np.newaxis], top_actions].A
            r_a = top_rewards.mean()
            train_m_rewards.append(r_a)
        
        print('Computing metrics on val')
        max_perf_index, max_perf = test_policy(pi, index, val_contexts, val_rewards, 512)
        val_max_index[i], val_max[i] = max_perf_index, max_perf
        print('Argmax reward on the validation : INDEX %.4f, TRUE %.4f'%(max_perf_index, max_perf))
        
    return train_m_rewards, val_max_index, val_max, true_duration