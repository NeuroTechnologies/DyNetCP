import numpy as np
import os
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')

args = parser.parse_args()

data_dir = args.data_dir
os.makedirs(data_dir, exist_ok=True)


# "Easy" dataset: 200 neurons, independent triplets

np.random.seed(3)
spike_rates = np.random.uniform(low=0.002, high=0.01, size=200)
num_triplets = 50
all_spikes = []
gt_pairs = []
gt_is_middle = []
gt_delays = []
triplet_types = []
triplet_is_middle = []
for n_idx in range(0, num_triplets*3, 3):
    if np.random.random() > 0.5:
        gt_is_middle.append(True)
        gt_is_middle.append(True)
        triplet_is_middle.append(True)
    else:
        gt_is_middle.append(False)
        gt_is_middle.append(False)
        triplet_is_middle.append(False)
    if np.random.random() > 0.5: #the "chain"
        triplet_types.append(True)
        gt_pairs.append([n_idx, n_idx+1])
        gt_pairs.append([n_idx+1, n_idx+2])
        gt_delays.append(2)
        gt_delays.append(1)
    else:
        triplet_types.append(False)
        gt_pairs.append([n_idx, n_idx+1])
        gt_pairs.append([n_idx, n_idx+2])
        gt_delays.append(2)
        gt_delays.append(1)
for trial in range(600):
    trial_spikes = []
    for triplet_idx in range(num_triplets):
        n_idx = 3*triplet_idx
        if gt_is_middle[triplet_idx]:
            n0_spikes = np.random.binomial(1, spike_rates[n_idx], size=(500,))
            connected_spikes_1 = np.random.binomial(1, 0.6, size=(500,))*n0_spikes
            n1_spikes = np.clip(
                np.random.binomial(1, spike_rates[n_idx+1], size=(500,)) +
                np.concatenate([np.zeros(2), connected_spikes_1[:-2]]),
                0, 1
            )
            other_n1_spikes_1 = np.random.binomial(1, spike_rates[n_idx+1], size=(125,))
            other_n1_spikes_2 = np.random.binomial(1, spike_rates[n_idx+1], size=(125,))
            n1_spikes = np.concatenate([other_n1_spikes_1, n1_spikes[125:375], other_n1_spikes_2])

            if triplet_types[triplet_idx]: #"chain"
                connected_spikes_2 = np.random.binomial(1, 0.6, size=(500,))*n1_spikes
            else:
                connected_spikes_2 = np.random.binomial(1, 0.6, size=(500,))*n0_spikes
            n2_spikes = np.clip(
                np.random.binomial(1, spike_rates[n_idx+2], size=(500,)) +
                np.concatenate([np.zeros(1), connected_spikes_2[:-1]]),
                0, 1
            )
            other_n2_spikes_1 = np.random.binomial(1, spike_rates[n_idx+2], size=(125,))
            other_n2_spikes_2 = np.random.binomial(1, spike_rates[n_idx+2], size=(125,))
            n2_spikes = np.concatenate([other_n2_spikes_1, n2_spikes[125:375], other_n2_spikes_2])
        else:
            n0_spikes = np.random.binomial(1, spike_rates[n_idx], size=(500,))
            connected_spikes_1 = np.random.binomial(1, 0.6, size=(500,))*n0_spikes
            n1_spikes = np.clip(
                np.random.binomial(1, spike_rates[n_idx+1], size=(500,)) +
                np.concatenate([np.zeros(2), connected_spikes_1[:-2]]),
                0, 1
            )
            other_n1_spikes = np.random.binomial(1, spike_rates[n_idx+1], size=(250,))
            n1_spikes = np.concatenate([n1_spikes[:125], other_n1_spikes, n1_spikes[-125:]])

            if triplet_types[triplet_idx]:
                connected_spikes_2 = np.random.binomial(1, 0.6, size=(500,))*n1_spikes
            else:
                connected_spikes_2 = np.random.binomial(1, 0.6, size=(500,))*n0_spikes
            n2_spikes = np.clip(
                np.random.binomial(1, spike_rates[n_idx+2], size=(500,)) +
                np.concatenate([np.zeros(1), connected_spikes_2[:-1]]),
                0, 1
            )
            other_n2_spikes = np.random.binomial(1, spike_rates[n_idx+2], size=(250,))
            n2_spikes = np.concatenate([n2_spikes[:125], other_n2_spikes, n2_spikes[-125:]])

        trial_spikes.extend([n0_spikes, n1_spikes, n2_spikes])
    for n, rate in enumerate(spike_rates[num_triplets*3:], num_triplets*3):
        trial_spikes.append(np.random.binomial(1, rate, size=(500,)))
    trial_spikes = np.stack(trial_spikes, axis=1)
    all_spikes.append(trial_spikes)
all_spikes = np.stack(all_spikes)
train_spikes = all_spikes[:400]
val_spikes = all_spikes[400:]

out_path = os.path.join(data_dir, 'synth_spikes_easy.npy')
np.save(out_path, [train_spikes, val_spikes, val_spikes])
gt_pairs_path = os.path.join(data_dir, 'pairs_easy.npy')
np.save(gt_pairs_path, gt_pairs)
gt_pairs_path = os.path.join(data_dir, 'is_middle_easy.npy')
np.save(gt_pairs_path, gt_is_middle)


# "medium" dataset: 100n, 50 random triplets (non independent)
np.random.seed(3)
num_triplets = 50
num_neurons = 100
num_trials = 600
connection_rate = 0.6


spike_rates = np.random.uniform(low=0.002, high=0.01, size=num_neurons)

all_spikes = []
gt_pairs = []
gt_is_middle = []
gt_delays = []
triplet_types = []
triplet_is_middle = []
range1 = np.arange(0, num_neurons // 3)
range2 = np.arange(num_neurons // 3, 2*num_neurons // 3)
range3 = np.arange(2*num_neurons // 3, num_neurons)
while len(triplet_types) < num_triplets:
    #n1, n2, n3 = sorted(np.random.choice(np.arange(num_neurons), size=3, replace=False))
    
    if np.random.random() > 0.5: #the "chain"
        triplet_types.append(True)
        n1 = np.random.choice(range1)
        n2 = np.random.choice(range2)
        n3 = np.random.choice(range3)
        gt_pairs.append([n1, n2])
        gt_pairs.append([n2, n3])
        gt_delays.append(2)
        gt_delays.append(1)
    else:
        triplet_types.append(False)
        n1 = np.random.choice(np.concatenate([range1, range2]))
        if n1 in range1:
            n2, n3 = np.random.choice(np.concatenate([range2, range3]), size=2, replace=False)
        else:
            n2, n3 = np.random.choice(range3, size=2, replace=False)
        gt_pairs.append([n1, n2])
        gt_pairs.append([n1, n3])
        gt_delays.append(2)
        gt_delays.append(1)
    if np.random.random() > 0.5:
        gt_is_middle.append(True)
        gt_is_middle.append(True)
        triplet_is_middle.append(True)
    else:
        gt_is_middle.append(False)
        gt_is_middle.append(False)
        triplet_is_middle.append(False)
all_spikes = []
for n, spr in enumerate(spike_rates):
    all_spikes.append(np.random.binomial(1, spr, size=(num_trials, 500,)))
all_spikes = np.stack(all_spikes, axis=-1)
for time in tqdm(range(500)):
    for pair_idx in range(len(gt_pairs)):
        n1, n2 = gt_pairs[pair_idx]
        delay = gt_delays[pair_idx]
        pair_is_middle = gt_is_middle[pair_idx]
        is_middle = time-delay >= 125 and time+delay < 375
        is_outside = time + delay < 125 or time - delay >= 375
        if time < delay or not ((is_middle and pair_is_middle) or (is_outside and not pair_is_middle)):
            continue
        connected_spikes = all_spikes[:, time-delay, n1] * np.random.binomial(1, connection_rate, size=(num_trials))
        all_spikes[:, time, n2] += connected_spikes
        all_spikes[:, time, n2] = np.clip(all_spikes[:, time, n2], 0, 1)
        

train_spikes = all_spikes[:400]
val_spikes = all_spikes[400:]

name = 'med'

out_path = os.path.join(data_dir, 'synth_spikes_%s.npy'%(name))
np.save(out_path, [train_spikes, val_spikes, val_spikes])
gt_pairs_path = os.path.join(data_dir, 'pairs_%s.npy'%(name))
np.save(gt_pairs_path, gt_pairs)
gt_pairs_path = os.path.join(data_dir, 'is_middle_%s.npy'%(name))
np.save(gt_pairs_path, gt_is_middle)



# "Hard": 100 n, 100 triplets
np.random.seed(3)
num_triplets = 50
num_neurons = 100
num_trials = 600
connection_rate = 0.6


spike_rates = np.random.uniform(low=0.002, high=0.01, size=num_neurons)

all_spikes = []
gt_pairs = []
gt_is_middle = []
gt_delays = []
triplet_types = []
triplet_is_middle = []
range1 = np.arange(0, num_neurons // 3)
range2 = np.arange(num_neurons // 3, 2*num_neurons // 3)
range3 = np.arange(2*num_neurons // 3, num_neurons)
while len(triplet_types) < num_triplets:
    #n1, n2, n3 = sorted(np.random.choice(np.arange(num_neurons), size=3, replace=False))
    
    if np.random.random() > 0.5: #the "chain"
        triplet_types.append(True)
        n1 = np.random.choice(range1)
        n2 = np.random.choice(range2)
        n3 = np.random.choice(range3)
        gt_pairs.append([n1, n2])
        gt_pairs.append([n2, n3])
        gt_delays.append(2)
        gt_delays.append(1)
    else:
        triplet_types.append(False)
        n1 = np.random.choice(np.concatenate([range1, range2]))
        if n1 in range1:
            n2, n3 = np.random.choice(np.concatenate([range2, range3]), size=2, replace=False)
        else:
            n2, n3 = np.random.choice(range3, size=2, replace=False)
        gt_pairs.append([n1, n2])
        gt_pairs.append([n1, n3])
        gt_delays.append(2)
        gt_delays.append(1)
    if np.random.random() > 0.5:
        gt_is_middle.append(True)
        gt_is_middle.append(True)
        triplet_is_middle.append(True)
    else:
        gt_is_middle.append(False)
        gt_is_middle.append(False)
        triplet_is_middle.append(False)
all_spikes = []
for n, spr in enumerate(spike_rates):
    all_spikes.append(np.random.binomial(1, spr, size=(num_trials, 500,)))
all_spikes = np.stack(all_spikes, axis=-1)
for time in tqdm(range(500)):
    for pair_idx in range(len(gt_pairs)):
        n1, n2 = gt_pairs[pair_idx]
        delay = gt_delays[pair_idx]
        pair_is_middle = gt_is_middle[pair_idx]
        is_middle = time-delay >= 125 and time+delay < 375
        is_outside = time + delay < 125 or time - delay >= 375
        if time < delay or not ((is_middle and pair_is_middle) or (is_outside and not pair_is_middle)):
            continue
        connected_spikes = all_spikes[:, time-delay, n1] * np.random.binomial(1, connection_rate, size=(num_trials))
        all_spikes[:, time, n2] += connected_spikes
        all_spikes[:, time, n2] = np.clip(all_spikes[:, time, n2], 0, 1)
        

train_spikes = all_spikes[:400]
val_spikes = all_spikes[400:]

name = 'hard'

out_path = os.path.join(data_dir, 'synth_spikes_%s.npy'%(name))
np.save(out_path, [train_spikes, val_spikes, val_spikes])
gt_pairs_path = os.path.join(data_dir, 'pairs_%s.npy'%(name))
np.save(gt_pairs_path, gt_pairs)
gt_pairs_path = os.path.join(data_dir, 'is_middle_%s.npy'%(name))
np.save(gt_pairs_path, gt_is_middle)