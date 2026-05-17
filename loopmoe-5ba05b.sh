export MODELSCOPE_CACHE=/volume/pt-train/users/ztgao/cache
export MODELSCOPE_MODULES_CACHE=/volume/pt-train/users/ztgao/cache
export MODELSCOPE_CACHE_DIR=/volume/pt-train/users/ztgao/cache
export NCCL_P2P_LEVEL=NVL

cd /volume/pt-train/users/ztgao/ms-swift
source activate
conda activate /volume/pt-train/users/ztgao/envs/swift
 
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
swift sft \
    --model /volume/pt-train/users/ztgao/workspace/Baseline-Normal-0p75alpha/ckpt/wrong_lr/iter_0410000/hf \
    --tuner_type full \
    --split_dataset_ratio 0.0001 \
    --save_safetensors true \
    --dataset '/volume/pt-train/users/ztgao/data/sft/Nemotron-Cascade-2-SFT-Data/mixed_unshuffled.jsonl' \
    --load_from_cache_file true \
    --expert_model_parallel_size 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --moe_aux_loss_coeff 0.01 \
    --learning_rate 1e-5 \
    --warmup_steps 1000 \
    --num_train_epochs 20 \
    --output_dir /volume/pt-train/users/ztgao/workspace/0p75alpha-sft \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length 128000 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --padding_free false \
    --template qwen3 \
    --loss_scale ignore_empty_think \
    --enable_loop \
    --loop_method layer-loop \
    --loop_times 2 \
    --eval_strategy "steps" \
    --eval_steps "5" \
    --per_device_eval_batch_size "5" \
    --eval_use_evalscope \
    --eval_dataset openai_mrcr needle_haystack aime24 aime25 aime26 amc arc arena_hard bbh drop gpqa_diamond gsm8k hellaswag hle hmmt25 humaneval humaneval_plus ifeval logi_qa math_500 mbpp mbpp_plus minerva_math mmlu mmlu_pro mmlu_redux sciq simple_qa super_gpqa swe_bench_verified terminal_bench_v2 trivia_qa winogrande \
    --eval_limit "10"