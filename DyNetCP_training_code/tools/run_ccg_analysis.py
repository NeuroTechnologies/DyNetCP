import os
import argparse

import numpy as np
import numpy as np
from scipy import fftpack
import math
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import argparse

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")



def get_units(session, snr_thresh, structure_list, stimuli, spike_count_threshold=0):
    
    stimulus_presentation_ids = (session.stimulus_presentations[
                                session.stimulus_presentations['stimulus_name'] == stimuli].index.values)
    decent_snr_units = session.units[(session.units['snr']>=snr_thresh) &
                                     (session.units['ecephys_structure_acronym'].isin(structure_list))]
    decent_snr_unit_ids = decent_snr_units.index.values
    time_bin_edges = np.arange(0.05,0.50005,0.001)
    spike_counts_da = session.presentationwise_spike_counts(bin_edges = time_bin_edges,
                                                   stimulus_presentation_ids=stimulus_presentation_ids,
                                                   unit_ids=decent_snr_unit_ids)
    mean_rates = spike_counts_da.mean(dim='stimulus_presentation_id').mean('time_relative_to_stimulus_onset')*1000
    filtered_unit_ids = mean_rates[mean_rates > 2].unit_id.data
    filtered_units = decent_snr_units.loc[filtered_unit_ids]
    return filtered_unit_ids, filtered_units, stimulus_presentation_ids


def nextpow2(x):
    if x == 0:
        y = 0
    else:
        y = math.ceil(math.log2(x))
    return y

# This implementation taken from https://stackoverflow.com/a/60245667
def xcorr(x, y, maxlag):
    m = max(len(x), len(y))
    mx1 = min(maxlag, m - 1)
    ceilLog2 = nextpow2(2 * m - 1)
    m2 = 2 ** ceilLog2

    X = fftpack.fft(x, m2)
    Y = fftpack.fft(y, m2)
    c1 = np.real(fftpack.ifft(X * np.conj(Y)))
    index1 = np.arange(1, mx1+1, 1) + (m2 - mx1 -1)
    index2 = np.arange(1, mx1+2, 1) - 1
    c = np.hstack((c1[index1], c1[index2]))
    return c


def compute_ccg(n1, n2, ccg_len=100, bin_size=0.001):
    #n1, n2 should be [ntrials, ntimesteps] binned trial data
    all_corr = np.zeros(2*ccg_len+1)
    M, t = n1.shape
    r1 = n1.mean()*1000
    r2 = n2.mean()*1000
    mid = t-1
    for trial in range(len(n1)):
        new_corr = xcorr(n1[trial], n2[trial], ccg_len)
        all_corr += new_corr
    #print("DIV FACTOR: ",M*np.sqrt(r1*r2))
    all_corr = all_corr/(M*np.sqrt(r1*r2))
    triang = t*bin_size - np.abs(np.arange(-ccg_len*bin_size, ccg_len*bin_size+bin_size/2, bin_size))
    all_corr /= triang
    return all_corr


def compute_shuffled_ccg(n1, n2, ccg_len=100):
    tmp_sp = np.stack([n1, n2], axis=-1).transpose(1,0,2)
    jittered_sp = jitter(tmp_sp, 25)
    j_sp_1 = jittered_sp[:, :, 0].T
    j_sp_2 = jittered_sp[:, :, 1].T
    return compute_ccg(j_sp_1, j_sp_2, ccg_len=ccg_len)


def jitter(data, l):
    """
    Jittering multidemntational logical data where 
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
    """
    #input shape should be [time, ntrials, num_neurons]
    if len(np.shape(data))>3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data,(np.shape(data)[0],np.shape(data)[1],len(data.flatten())/(np.shape(data)[0]*np.shape(data)[1])), order='F')
    else:
        flag = 0
    psth = np.mean(data,axis=1)
    length = np.shape(data)[0]

    # pad to be divisible by l
    if np.mod(np.shape(data)[0],l):
        data[length:(length+np.mod(-np.shape(data)[0],l)),:,:] = 0
        psth[length:(length+np.mod(-np.shape(data)[0],l)),:]   = 0

    #dataj and psthj sums over the window (size l)
    if np.shape(psth)[1]>1:
        dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0))
        psthj = np.squeeze(np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0))
    else:
        dataj = np.squeeze(np.sum(np.reshape(data,l,np.shape(data)[0]//l,np.shape(data)[1], order='F')))
        psthj = np.sum(np.reshape(psth,l,np.shape(psth)[0]//l, order='F'))
    
    
    # this ensures that we didn't squeeze out the first dim (if time == l)
    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj,[1,np.shape(dataj)[0],np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj,[1,np.shape(psthj[0])], order='F');

    psthj = np.reshape(psthj,[np.shape(psthj)[0],1,np.shape(psthj)[1]], order='F')
    psthj[psthj==0] = 10e-10

    # corr is the number of spikes in each window (per trial, per coarse trial time)
    #    divided by the spike rate (averaged over trials) summed over the window
    corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1], 1]);
    corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1],np.shape(corr)[2]], order='F')
    corr = np.tile(corr,[l, 1, 1, 1])
    corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2],np.shape(corr)[3]], order='F');

    psth = np.reshape(psth,[np.shape(psth)[0],1,np.shape(psth)[1]], order='F');
    # this final output multiplies corr by the spike rate at every point in time
    output = np.tile(psth,[1, np.shape(corr)[1], 1])*corr

    output = output[:length,:,:]
    return output


def get_clean_units(session, clean_units, structure_list, stimuli_list, spike_count_threshold=0):
    clean_units = clean_units.index.values
    if stimuli_list is not None:
        if not isinstance(stimuli_list, list):
            stimuli_list = list(stimuli_list)
        stimulus_presentation_ids = (session.stimulus_presentations[
                                    session.stimulus_presentations['stimulus_name'].isin(stimuli_list)].index.values)
    else:
        stimulus_presentation_ids = session.stimulus_presentations.index.values
    decent_snr_units = session.units[(session.units.index.isin(clean_units)) &
                                     (session.units['ecephys_structure_acronym'].isin(structure_list))]
    decent_snr_unit_ids = decent_snr_units.index.values
    time_bin_edges = np.arange(0.05,0.50005,0.001)
    print(len(time_bin_edges))
    spike_counts_da = session.presentationwise_spike_counts(bin_edges = time_bin_edges,
                                                   stimulus_presentation_ids=stimulus_presentation_ids,
                                                   unit_ids=decent_snr_unit_ids)
    mean_rates = spike_counts_da.mean(dim='stimulus_presentation_id').mean('time_relative_to_stimulus_onset')*1000
    
    filtered_unit_ids = mean_rates[mean_rates > 2].unit_id.data
    filtered_units = decent_snr_units.loc[filtered_unit_ids]
    return filtered_unit_ids, filtered_units, stimulus_presentation_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory')
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--working_dir')
    args = parser.parse_args()
    data_directory = args.data_directory
    session_id = args.session_id
    working_dir = args.working_dir

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    manifest_path = os.path.join(data_directory, 'cache', 'manifest.json')

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()

    session = cache.get_session_data(session_id,
                                     isi_violations_maximum = np.inf,
                                     amplitude_cutoff_maximum = np.inf,
                                     presence_ratio_minimum = -np.inf
                                    )
    units = cache.get_units()
    filtered_units = units[(units['amplitude_cutoff'] < 0.1)&\
      (units['isi_violations']<0.1)&\
      (units['presence_ratio']>0.95)&\
      (units['isolation_distance']>30)&\
      (units['snr']>1)]
    stimuli = ['drifting_gratings_75_repeats']
    unit_ids, units, stim_pres_ids = get_clean_units(session, filtered_units, ["VISp","VISl","VISrl","VISal","VISpm","VISam"], stimuli)


    time_bin_edges = np.arange(-0.150, 2.0005, 0.001)
    spike_counts = session.presentationwise_spike_counts(bin_edges=time_bin_edges,
                                                         stimulus_presentation_ids=stim_pres_ids,
                                                         unit_ids=unit_ids)
    spikes = np.array(spike_counts).astype(float).clip(0,1)

    count = 0
    all_sig = []
    all_offsets = []
    offset_path = os.path.join(working_dir, 'offsets.npy')
    sig_path = os.path.join(working_dir, 'all_sig.npy')
    prog_path = os.path.join(working_dir, 'progress.txt')
    for n1, n1_id in enumerate(unit_ids):
        print("Processing %d of %d"%(n1+1, len(unit_ids)))
        u1 = units.loc[n1_id]
        sp1 = spikes[:, :, n1]
        for n2, n2_id in tqdm(enumerate(unit_ids[n1+1:]), total=len(unit_ids[n1+1:])):
            n2 = n1 + n2 + 1
            u2 = units.loc[n2_id]
            sp2 = spikes[:, :, n2]
            ccg = compute_ccg(sp1, sp2)
            shuffle_ccg = compute_shuffled_ccg(sp1, sp2)
            corrected_ccg = ccg - shuffle_ccg
            ends = np.concatenate([corrected_ccg[:50], corrected_ccg[-51:]])
            std = ends.std().item()
            mid = 100
            peak = corrected_ccg[mid-10:mid+11].max().item()
            if peak > 7*std:
                ind = corrected_ccg[mid-10:mid+11].argmax().item()-10
                all_offsets.append(ind)
                all_sig.append(((n1, n1_id), (n2, n2_id)))
        print("Done. Current found: ",len(all_sig))
        print()
        np.save(offset_path, np.array(all_offsets))
        np.save(sig_path, all_sig)
        with open(prog_path, 'w') as fout:
            fout.write('%d'%n1)
    print("FINAL FOUND: ",len(all_sig))
    print("ALL OFFSETS: ",all_offsets)
    print("ALL SIG: ",all_sig)

        