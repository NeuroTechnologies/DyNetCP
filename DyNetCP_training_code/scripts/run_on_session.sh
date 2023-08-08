gpu=0

session=771160300
base_data_dir=data/
data_dir=${base_data_dir}processed_data/
exp_dir=experiments/

echo "Preparing data for session ${sess}..."
python -u tools/create_dataset.py --data_directory $base_data_dir --session_id $session

############################################
#### STEP 1: train uncorrected static, bin 1
##############################################
sess_dir_1ms="${data_dir}${session}/bin1/"
id=$session
echo "RUNNING SESS: ${id}"

data_path_1ms="${sess_dir_1ms}spikes_all.h5"
working_dir_1ms="${exp_dir}${id}/bin1/deconly_delay100/"
mkdir -p $working_dir_1ms
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --continue_training \
    --config_file ./configs/dsap_deconly.yaml \
    --extra_args data.spikes_path $data_path_1ms \
    --extra_args data.dataset_type h5 \
    --extra_args num_edge_types 100 \
    --extra_args model.l2_coef 100.0 \
    --extra_args data.use_jitter_correction False \
    --extra_args model.train_with_correction False \
    --extra_args model.val_with_correction False \
    --working_dir $working_dir_1ms |& tee "${working_dir_1ms}results.txt"
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
    --load_best_model \
    --working_dir $working_dir_1ms



############################################
#### STEP 2: train corrected static, bin 1
##############################################
corrected_1ms_working_dir="${exp_dir}${id}/bin1/correctedonly_deconly_delay100/"
mkdir -p $corrected_1ms_working_dir
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --continue_training \
    --config_file ./configs/dsap_deconly.yaml \
    --extra_args data.spikes_path $data_path_1ms \
    --extra_args data.dataset_type h5 \
    --extra_args num_edge_types 100 \
    --extra_args model.l2_coef 100.0 \
    --extra_args training.early_stopping_iters 100 \
    --extra_args data.use_jitter_correction True \
    --extra_args data.jitter_correction_window 25 \
    --extra_args model.train_with_correction True \
    --extra_args model.correction_only True \
    --extra_args model.corrected_nll_coef 1.0 \
    --extra_args model.val_with_correction True \
    --extra_args data.jitter_correction_type trial_exact \
    --working_dir $corrected_1ms_working_dir |& tee "${corrected_1ms_working_dir}results.txt" 
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
    --load_best_model \
    --working_dir $corrected_1ms_working_dir

##################################################################3
### STEP 3: determine pairs
#####################################################################
spthresh=2000
thresh=7
edge_name=thresholded_edges_std_${thresh}_spthresh${spthresh}
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/threshold_edges.py \
    --load_best_model \
    --extra_args data.spikes_path $data_path_1ms \
    --working_dir $working_dir_1ms \
    --correction_model_path ${corrected_1ms_working_dir}best_model \
    --std_threshold $thresh \
    --use_abs \
    --joint_spike_threshold $spthresh \
    --spikes_path $data_path_1ms \
    --name $edge_name

edge_name_bothdirs=thresholded_edges_std_${thresh}_bothdirs_spthresh${spthresh}
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/threshold_edges.py \
    --load_best_model \
    --extra_args data.spikes_path $data_path_1ms \
    --working_dir $working_dir_1ms \
    --correction_model_path ${corrected_1ms_working_dir}best_model \
    --std_threshold $thresh \
    --both_directions \
    --use_abs \
    --joint_spike_threshold $spthresh \
    --spikes_path $data_path_1ms \
    --name $edge_name_bothdirs
    
###################################
### STEP 4: Train Static model on 5ms binned data
####################################
sess_dir_5ms="${data_dir}${session}/bin5/"
data_path_5ms="${sess_dir_5ms}spikes_all.h5"
working_dir_5ms="${exp_dir}${id}/bin5/deconly_delay10/"
mkdir -p $working_dir_5ms
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --continue_training \
    --config_file ./configs/dsap_deconly.yaml \
    --extra_args data.spikes_path $data_path_5ms \
    --extra_args data.dataset_type h5 \
    --extra_args num_edge_types 10 \
    --extra_args model.l2_coef 10.0 \
    --working_dir $working_dir_5ms |& tee "${working_dir_5ms}results.txt"

#########################################
### STEP 5: Train dyn model on 5 ms binned data
##########################################
edge_path=${working_dir_1ms}${edge_name_bothdirs}.npy
new_working_dir="${working_dir_5ms}dyn_thresh${thresh}_spthresh${spthresh}/"
mkdir -p $new_working_dir
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --load_model ${working_dir_5ms}best_model \
    --extra_args model.edge_path $edge_path \
    --config_file ./configs/dsap_enconly.yaml \
    --extra_args data.dataset_type h5 \
    --extra_args num_edge_types 10 \
    --extra_args data.spikes_path $data_path_5ms \
    --extra_args model.l2_coef 10.0 \
    --extra_args data.max_time 150 \
    --extra_args training.accumulate_steps 1 \
    --extra_args training.batch_size 16 \
    --extra_args training.early_stopping_iters 50 \
    --continue_training \
    --working_dir $new_working_dir |& tee "${new_working_dir}results.txt"
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
    --load_best_model \
    --working_dir $new_working_dir 


####################################
## STEP 6: model on single pairs
####################################
edge_cnt_path=${working_dir_1ms}${edge_name}_len.txt
num_pairs=$(($(cat "$edge_cnt_path")-1))
echo "NUM PAIRS: ${num_pairs}"
for pair in $(seq 0 $num_pairs ); do
    edge_path=${working_dir_1ms}${edge_name}.npy
    working_dir_singlepair="${exp_dir}${id}/bin5/deconly_delay100/allsigpairs_widepeak/${pair}/"
    mkdir -p $working_dir_singlepair
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
        --continue_training \
        --config_file ./configs/dsap_deconly.yaml \
        --extra_args data.spikes_path $data_path_5ms \
        --extra_args data.dataset_type h5 \
        --extra_args num_edge_types 10 \
        --extra_args model.l2_coef 1.0 \
        --extra_args training.early_stopping_iters 100 \
        --extra_args data.pairs_file $edge_path \
        --extra_args data.pair_idx $pair \
        --working_dir $working_dir_singlepair |& tee "${working_dir_singlepair}results.txt"

    new_working_dir_singlepair="${working_dir_singlepair}dyn/"
    mkdir -p $new_working_dir_singlepair
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --load_model ${working_dir_singlepair}best_model \
    --extra_args model.edge_path $edge_path \
    --extra_args model.edge_ind $pair \
    --extra_args data.pair_idx $pair \
    --extra_args data.pairs_file $edge_path \
    --config_file ./configs/dsap_enconly.yaml \
    --extra_args data.dataset_type h5 \
    --extra_args num_edge_types 10 \
    --extra_args data.spikes_path $data_path_5ms \
    --extra_args model.l2_coef 1.0 \
    --extra_args data.max_time 150 \
    --extra_args training.accumulate_steps 1 \
    --extra_args training.batch_size 16 \
    --extra_args training.early_stopping_iters 50 \
    --continue_training \
    --working_dir $new_working_dir_singlepair |& tee "${new_working_dir_singlepair}results.txt"
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
        --load_best_model \
        --working_dir $new_working_dir_singlepair 
done