gpu=0

data_dir=processed_data/hh_synth/
exp_dir=experiments/hh_synth/

working_dir="${exp_dir}deconly/"
mkdir -p $working_dir
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --continue_training \
    --config_file ./configs/dsap_deconly.yaml \
    --extra_args data.data_dir $data_dir \
    --extra_args data.dataset_type hh \
    --extra_args data.is_small_data True \
    --extra_args data.trial_len 1000 \
    --extra_args num_edge_types 100 \
    --extra_args model.l2_coef 0.1 \
    --extra_args training.steps_per_epoch 100 \
    --extra_args data.use_jitter_correction False \
    --extra_args model.train_with_correction False \
    --extra_args model.val_with_correction False \
    --extra_args model.decoder.init_random_uniform 0.01 \
    --extra_args training.early_stopping_iters 100 \
    --working_dir $working_dir |& tee "${working_dir}results.txt"
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
    --load_best_model \
    --working_dir $working_dir

working_dir="${exp_dir}jittered_deconly/"
mkdir -p $working_dir
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/train_model.py \
    --continue_training \
    --config_file ./configs/dsap_deconly.yaml \
    --extra_args data.data_dir $data_dir \
    --extra_args data.dataset_type hh \
    --extra_args data.trial_len 1000 \
    --extra_args num_edge_types 100 \
    --extra_args model.l2_coef 0.1 \
    --extra_args training.steps_per_epoch 100 \
    --extra_args data.is_small_data True \
    --extra_args data.use_jitter_correction False \
    --extra_args model.train_with_correction False \
    --extra_args model.val_with_correction False \
    --extra_args model.decoder.init_random_uniform 0.01 \
    --extra_args training.early_stopping_iters 100 \
    --extra_args data.use_jitter_correction True \
    --extra_args model.train_with_correction True \
    --extra_args model.correction_only True \
    --extra_args model.corrected_nll_coef 1.0 \
    --extra_args model.val_with_correction True \
    --extra_args data.jitter_correction_window 5 \
    --extra_args data.jitter_correction_type uniform_window \
    --working_dir $working_dir |& tee "${working_dir}results.txt"
CUDA_VISIBLE_DEVICES=$gpu python -u dsap/experiments/save_weights.py \
    --load_best_model \
    --working_dir $working_dir