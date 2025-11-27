"""
Created on Sun Aug  1 04:40:45 2021
Autoencoder
@author: mi
"""

import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import random
import math
import time
import segyio
from matplotlib import pyplot as plt


def kl_divergence(p, q):
    s1 = p * torch.log(p / q)
    s2 = (1 - p) * torch.log((1 - p) / (1 - q))
    output = torch.mean(s1 + s2)
    return output


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=900, hidden_size=1800):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            # nn.ReLU()
            # nn.Softplus()
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def next_batch_random(train_data, size_batch):
    index = [i for i in range(0, len(train_data))]
    np.random.shuffle(index)
    batch_data = []
    for i in range(0, size_batch):
        batch_data.append(train_data[index[i]])
    batch_data = np.array(batch_data)
    return batch_data


def patch_regular(input2d, l1, l2, s1, s2):
    n1 = input2d.shape[0]
    n2 = input2d.shape[1]
    output2d = []
    for i1 in range(0, n1 - l1 + 1, s1):
        for i2 in range(0, n2 - l2 + 1, s2):
            tmp = np.reshape(input2d[i1:i1 + l1, i2:i2 + l2], (l1 * l2, 1), order="F")
            output2d.append(tmp)
    output2d = np.array(output2d)
    output2d = np.squeeze(output2d, axis=(2,))
    return output2d


def patch_regular_inv(input2d, n1, n2, l1, l2, s1, s2):
    output2d = np.zeros((n1, n2))
    count2d = np.zeros((n1, n2))
    tmp = 0
    for i1 in range(0, n1 - l1 + 1, s1):
        for i2 in range(0, n2 - l2 + 1, s2):
            output2d[i1:i1 + l1, i2:i2 + l2] = output2d[i1:i1 + l1, i2:i2 + l2] + np.reshape(input2d[tmp, :], (l1, l2),
                                                                                             order="F")
            count2d[i1:i1 + l1, i2:i2 + l2] = count2d[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            tmp = tmp + 1
    output2d = output2d / count2d
    return output2d


def patch_random(input2d, patch_num, l1, l2):
    n1 = input2d.shape[0]
    n2 = input2d.shape[1]
    output2d = np.zeros((patch_num, l1 * l2))

    for i1 in range(0, patch_num, 1):
        choose_row = int(round((n1 - l1) * random.random()))
        choose_col = int(round((n2 - l2) * random.random()))
        tmp = input2d[choose_row:choose_row + l1, choose_col:choose_col + l2]
        output2d[i1, :] = np.reshape(tmp, l1 * l2, order="F")
    return output2d


def min_matrix(input2d):
    output2d = []
    for i in range(len(input2d)):
        output2d.append(min(input2d[i]))
    return min(output2d)


def max_matrix(input2d):
    output2d = []
    for i in range(len(input2d)):
        output2d.append(max(input2d[i]))
    return max(output2d)

clean_data = np.fromfile("Syn_prestack_128_128.dat", dtype=np.float64)
clean_data = clean_data.reshape(128, 128)
# (noise_data1, noise, snr) = Gnoisegen(clean_data, 2)
noise_data1 = np.load('Syn_pre_noisy.npy')


min1 = min_matrix(noise_data1)
max1 = max_matrix(noise_data1)
d_normal = (noise_data1 - min1) / (max1 - min1)

# generate patches for training and testing sets
patch1 = 32
patch2 = 32
sliding1 = 2
sliding2 = 2
n_patch = 14000
n_hidden = patch1*patch2 + 1000
D_train = patch_random(d_normal, n_patch, patch1, patch2)
D_test = patch_regular(d_normal, patch1, patch2, sliding1, sliding2)

n_samples = len(D_train)


batch_size = 20
num_epochs = 20
learning_rate = 0.001
sparse_reg = 0.05
expect_tho = 0.02

autoEncoder = AutoEncoder(in_dim=patch1*patch2, hidden_size=n_hidden)

GPU = True
if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if torch.cuda.is_available():
    autoEncoder.cuda()

optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=learning_rate)
loss_func1 = nn.MSELoss()

d_test = torch.tensor(D_test).to(device, dtype=torch.float32)
for epoch in range(num_epochs):
    time_epoch_start = time.time()
    total_batch = int(n_samples / batch_size)
    avg_cost = 0
    for batch_index in range(total_batch):
        batch_xs = next_batch_random(D_train, batch_size)
        if torch.cuda.is_available():
            batch_xs = torch.Tensor(batch_xs).float().cuda()

        encoder_out, decoder_out = autoEncoder(batch_xs)
        loss_mse = loss_func1(decoder_out, batch_xs)
        loss_kl = kl_divergence(expect_tho, encoder_out[-1])
        loss = loss_mse + sparse_reg * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_cost += loss / n_samples * batch_size
    print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, avg_cost, time.time() - time_epoch_start))
print('slice of %d in 1000 is done' % num_epochs)

_, d_recon = autoEncoder(d_test)
d_recon_cpu = d_recon.cuda().data.cpu().numpy()
d_final = patch_regular_inv(d_recon_cpu, d_normal.shape[0], d_normal.shape[1], patch1, patch2, sliding1, sliding2)
d_final = d_final * (max1 - min1) + min1
d_final = d_final.astype(np.float32)

noise_residual=clean_data-d_final

s = math.pow(np.linalg.norm(clean_data, ord=2), 2)
n = math.pow(np.linalg.norm(clean_data - d_final, ord=2), 2)
denoised_SNR = 10 * np.log10(s / n)

plt.figure('denoised_SNR:' + str(round(denoised_SNR, 2)))
plt.imshow(d_final, aspect='auto', cmap='gray')
plt.colorbar()
plt.savefig('Syn_prestack_128_128_sae_' + str(
    round(denoised_SNR, 2)) + '__' + '_denoised.png')
