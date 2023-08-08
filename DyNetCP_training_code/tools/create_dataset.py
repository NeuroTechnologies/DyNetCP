import os, subprocess

import numpy as np

import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")
import pandas as pd
import argparse
import h5py
import itertools


def get_clean_units(session, clean_units, structure_list, stimuli_list,
                    spike_rate_threshold=-55):
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
    spike_counts_da = session.presentationwise_spike_counts(bin_edges = time_bin_edges,
                                                   stimulus_presentation_ids=stimulus_presentation_ids,
                                                   unit_ids=decent_snr_unit_ids)
    mean_rates = spike_counts_da.mean(dim='stimulus_presentation_id').mean('time_relative_to_stimulus_onset')*1000
    
    filtered_unit_ids = mean_rates[mean_rates > spike_rate_threshold].unit_id.data
    filtered_units = decent_snr_units.loc[filtered_unit_ids]
    return filtered_unit_ids, filtered_units, stimulus_presentation_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory')
    parser.add_argument('--session_id', type=int)

    args = parser.parse_args()
    data_directory = args.data_directory
    session_id = args.session_id

    manifest_path = os.path.join(data_directory, 'cache', 'manifest.json')
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()

    units = cache.get_units()
    filtered_units = units[(units['amplitude_cutoff'] < 0.1)&\
      (units['isi_violations']<0.1)&\
      (units['presence_ratio']>0.95)&\
      (units['isolation_distance']>30)&\
      (units['snr']>1)]

    sess = sessions.loc[session_id]
    session_thresh = cache.get_session_data(session_id,
                                            isi_violations_maximum=np.inf,
                                            amplitude_cutoff_maximum=np.inf,
                                            presence_ratio_minimum=-np.inf,
                                           )
    stimuli = ['drifting_gratings_75_repeats']
    thresh_unit_ids, thresh_units, thresh_stim_pres_ids = get_clean_units(session_thresh, filtered_units, 
                                                                          ["VISp","VISl","VISrl","VISal","VISpm","VISam"], 
                                                                          stimuli, spike_rate_threshold=-55)
    
    
    time_options = [
        np.arange(-0.150, 2.0005, 0.001),
        np.arange(-0.150, 2.0005, 0.005),
    ]
    out_dir_options = [
        os.path.join(data_directory, 'processed_data', str(session_id), 'bin1'),
        os.path.join(data_directory, 'processed_data', str(session_id), 'bin5')
    ]
    for time_bin_edges, out_dir in zip(time_options, out_dir_options):
        os.makedirs(out_dir, exist_ok=True)
    
    
        orientations = session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids]['orientation'].unique()
        blocks = session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids]['stimulus_block'].unique()
        sessions_by_orientation_running = []
        sessions_by_orientation_notrunning = []
        spikes_by_orientation_running = []
        spikes_by_orientation_notrunning = []
        running_by_orientation = []
        
        stims = session_thresh.stimulus_presentations.loc[thresh_stim_pres_ids]
        stim_time_bins = stims['start_time'][:, None] + time_bin_edges
        run_result = np.searchsorted(session_thresh.running_speed["start_time"], stim_time_bins)-1
        final_running_speeds = session_thresh.running_speed['velocity'].iloc[run_result.reshape(-1)].values.reshape(run_result.shape)
        run_bool = final_running_speeds.mean(1) > 5
        
        for orientation, block in itertools.product(orientations, blocks):
            valid = (session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids]['orientation'] == orientation) & \
                    (session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids]['stimulus_block'] == block)
            running = run_bool[valid]
            not_running = ~running
            o_run_stim_pres = session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids][valid][running]
            o_norun_stim_pres = session_thresh.stimulus_presentations.iloc[thresh_stim_pres_ids][valid][not_running]
            sessions_by_orientation_running.append(o_run_stim_pres)
            sessions_by_orientation_notrunning.append(o_norun_stim_pres)
            run_stim_ids = o_run_stim_pres.index.values
            norun_stim_ids = o_norun_stim_pres.index.values
            run_spike_counts = session_thresh.presentationwise_spike_counts(bin_edges=time_bin_edges,
                                                                stimulus_presentation_ids=run_stim_ids,
                                                                unit_ids=thresh_unit_ids)
            norun_spike_counts = session_thresh.presentationwise_spike_counts(bin_edges=time_bin_edges,
                                                                stimulus_presentation_ids=norun_stim_ids,
                                                                unit_ids=thresh_unit_ids)
            spikes_by_orientation_running.append(run_spike_counts)
            spikes_by_orientation_notrunning.append(norun_spike_counts)
        all_results = []
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_units_path = os.path.join(out_dir, 'units.pkl')
        thresh_units.to_pickle(out_units_path)
    
        all_train_run = []
        all_val_run = []
        all_train_norun = []
        all_val_norun = []
        
        
        for o_idx, spikes in enumerate(spikes_by_orientation_running):
            spikes = np.array(spikes).astype(float).clip(0,1)
            val_idx = int(len(spikes)*0.8)
            train_spikes, val_spikes = spikes[:val_idx], spikes[val_idx:]
            all_train_run.append(train_spikes)
            all_val_run.append(val_spikes)
            
        for o_idx, spikes in enumerate(spikes_by_orientation_notrunning):
            spikes = np.array(spikes).astype(float).clip(0,1)
            val_idx = int(len(spikes)*0.8)
            train_spikes, val_spikes = spikes[:val_idx], spikes[val_idx:]
            all_train_norun.append(train_spikes)
            all_val_norun.append(val_spikes)
            
            
        all_train_run = np.concatenate(all_train_run)
        all_val_run = np.concatenate(all_val_run)
        path = os.path.join(out_dir, 'spikes_all_run.h5')
        with h5py.File(path, 'w') as fout:
            fout.create_dataset('train', data=all_train_run)
            fout.create_dataset('val', data=all_val_run)
        all_train_norun = np.concatenate(all_train_norun)
        all_val_norun = np.concatenate(all_val_norun)
        path = os.path.join(out_dir, 'spikes_all_norun.h5')
        with h5py.File(path, 'w') as fout:
            fout.create_dataset('train', data=all_train_norun)
            fout.create_dataset('val', data=all_val_norun)
        
        all_train = np.concatenate([all_train_run, all_train_norun])
        all_val = np.concatenate([all_val_run, all_val_norun])
        path = os.path.join(out_dir, 'spikes_all.h5')
        with h5py.File(path, 'w') as fout:
            fout.create_dataset('train', data=all_train)
            fout.create_dataset('val', data=all_val)