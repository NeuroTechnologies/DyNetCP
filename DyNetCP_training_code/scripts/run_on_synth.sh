gpu=0

data_dir=data/processed_data/synth/
exp_dir=experiments/synth/

echo "Preparing synth data..."
python -u tools/generate_synth_data.py --data_dir $data_dir


names=( 'easy' 'medium' 'hard' )
for name in "${names[@]}"; do
    data_path=${data_dir}synth_spikes_${name}.npy
    working_dir_del100="${exp_dir}${name}/deconly_delay100/"
    mkdir -p $working_dir_del100
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
        --config_file ./configs/dsap_deconly.yaml \
        --extra_args data.spikes_path $data_path \
        --extra_args num_edge_types 100 \
        --extra_args model.l2_coef 100.0 \
        --extra_args training.early_stopping_iters 100 \
        --continue_training \
        --working_dir $working_dir_del100 |& tee "${working_dir_del100}results.txt"
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
        --load_best_model \
        --working_dir $working_dir_del100

    corrected_working_dir_del100="${exp_dir}${name}/correctedonly_deconly_delay100/"
    mkdir -p $corrected_working_dir_del100
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
        --config_file ./configs/dsap_deconly.yaml \
        --extra_args data.spikes_path $data_path \
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
        --continue_training \
        --working_dir $corrected_working_dir_del100 |& tee "${corrected_working_dir_del100}results.txt"
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
        --load_best_model \
        --working_dir $corrected_working_dir_del100



    thresh=7
    edge_name=thresholded_edges_std_${thresh}
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/threshold_edges.py \
        --load_best_model \
        --working_dir $working_dir_del100 \
        --correction_model_path ${corrected_working_dir_del100}best_model \
        --std_threshold $thresh \
        --use_abs \
        --name $edge_name
    
    edge_name_bothdirs=thresholded_edges_std_${thresh}_bothdirs
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/threshold_edges.py \
        --load_best_model \
        --working_dir $working_dir_del100 \
        --correction_model_path ${corrected_working_dir_del100}best_model \
        --std_threshold $thresh \
        --both_directions \
        --use_abs \
        --name $edge_name_bothdirs

    working_dir="${exp_dir}${name}/deconly_delay10/"
    mkdir -p $working_dir
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
        --config_file ./configs/dsap_deconly.yaml \
        --extra_args data.spikes_path $data_path \
        --extra_args num_edge_types 10 \
        --extra_args model.l2_coef 10.0 \
        --extra_args training.early_stopping_iters 100 \
        --continue_training \
        --working_dir $working_dir |& tee "${working_dir}results.txt"
    

    edge_path=${working_dir_del100}${edge_name_bothdirs}.npy
    new_working_dir="${working_dir}dyn_thresh${thresh}_del100/"
    mkdir -p $new_working_dir
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
        --load_model ${working_dir}best_model \
        --extra_args model.edge_path $edge_path \
        --config_file ./configs/dsap_enconly.yaml \
        --extra_args num_edge_types 10 \
        --extra_args data.spikes_path $data_path \
        --extra_args model.l2_coef 10.0 \
        --extra_args model.sparse_dynamic True \
        --extra_args training.accumulate_steps 4 \
        --extra_args training.batch_size 4 \
        --extra_args training.early_stopping_iters 50 \
        --continue_training \
        --working_dir $new_working_dir |& tee "${new_working_dir}results.txt"
    CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
        --load_best_model \
        --working_dir $new_working_dir 

done