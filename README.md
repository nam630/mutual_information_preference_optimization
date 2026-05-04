# Maximizing mutual information between prompts and responses improves LLM personalization with no additional data or human oversight (ICML'26)

<img width="1367" height="305" alt="diagram" src="https://github.com/user-attachments/assets/e2cc19e8-f878-4c6d-ad91-65f31e35acb5" />

[Arxiv link](https://arxiv.org/abs/2603.19294)

## Abstract

While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new high-quality data is expensive to collect. More fundamentally, true intelligence goes far beyond tasks that are easily verifiable. Therefore, we need self-improvement frameworks that allow models to improve without external oversight.

We propose **Mutual Information Preference Optimization (MIPO)**, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization (DPO) to learn from this paired data maximizes pointwise conditional mutual information (MI) (under the base LLM) between prompts and model responses.

Empirical results with various-sized Llama- and Qwen-Instruct models show that when used to maximize MI between user context and response, MIPO provides an effective personalization technique, achieving 3–40% improvements on personalization tasks using real-user datasets compared to strong baselines. Surprisingly, MIPO can also be applied to improve performance on math and multiple-choice problems, yielding 1–18% improvement *without any additional data or human supervision*. These results suggest a promising direction for self-improvement.


## Codebase

This codebase is modified from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

## Training Commands

Below are example commands for the main training methods.

### RLVR on GSM8K

```
deepspeed --module openrlhf.cli.train_self_play --pretrain Qwen/Qwen2.5-7B-Instruct --save_path [PATH NAME] --save_steps 100 --logging_steps 1 --eval_steps -1 --micro_train_batch_size 1 --train_batch_size 4 --micro_rollout_batch_size 2 --rollout_batch_size 8 --max_epochs 2 --prompt_max_len 15360 --generate_max_len 256 --zero_stage 2 --bf16 --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 --init_kl_coef 0.001 --prompt_data json@[JSONL FILE PATH] --apply_chat_template --max_samples 100000 --normalize_reward --flash_attn --input_key augmented_prompt --use_qwen

```

### MIPO (with DPO backbone)

```
deepspeed --module openrlhf.cli.train_dpo --save_path [PATH NAME] --save_steps -1 --logging_steps 1 --eval_steps -1 --train_batch_size 4 --micro_train_batch_size 1 --pretrain Qwen/Qwen2.5-3B-Instruct --bf16 --max_epochs 1 --max_len 15360 --zero_stage 2 --learning_rate 5e-7 --beta 0.1 --dataset json@[JSONL FILE PATH] --apply_chat_template --chosen_key chosen --rejected_key rejected --flash_attn --load_checkpoint --packing_samples --gradient_checkpointing 
```

### InfoNCE

```
deepspeed --module openrlhf.cli.train_infonce --save_path [PATH NAME] --save_steps -1 --logging_steps 1 --eval_steps -1 --train_batch_size 4 --micro_train_batch_size 1 --pretrain Qwen/Qwen2.5-3B-Instruct --bf16 --max_epochs 1 --max_len 15360 --zero_stage 2 --learning_rate 5e-7 --beta 0.1 --dataset json@[JSONL FILE PATH]  --apply_chat_template --chosen_key chosen --rejected_key rejected --flash_attn --load_checkpoint --packing_samples --gradient_checkpointing 
```

### SAMI

```
deepspeed --module openrlhf.cli.train_infonce --save_path [PATH NAME] --save_steps -1 --logging_steps 1 --eval_steps -1 --train_batch_size 4 --micro_train_batch_size 1 --pretrain Qwen/Qwen2.5-3B-Instruct --bf16 --max_epochs 1 --max_len 15360 --zero_stage 2 --learning_rate 5e-7 --beta 0.1 --dataset json@[JSONL FILE PATH]  --apply_chat_template --chosen_key chosen --rejected_key rejected --flash_attn --load_checkpoint --packing_samples --gradient_checkpointing  --sami
```

### SFT

```
deepspeed --module openrlhf.cli.train_sft   --max_len 15360   --dataset json@[JSONL FILE PATH   --input_key question    --output_key response    --train_batch_size 4    --micro_train_batch_size 1    --max_samples 500000    --pretrain Qwen/Qwen2.5-3B-Instruct --save_path [PATH NAME] --save_steps -1    --logging_steps 1    --eval_steps -1    --zero_stage 2    --max_epochs 1    --packing_samples    --bf16    --learning_rate 5e-7    --gradient_checkpointing    --use_wandb 173ea73581c8f47fb8aa18a4b33040f44e4ebdd6
```

## Reference
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```
