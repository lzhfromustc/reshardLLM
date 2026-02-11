# script parameters:
# 0: conversation_mode (first/all/all-wait)
# 1: qps, 2: query_distribution, 3: coefficient_variation
# 4: imean, 5: omean
# 6: dispatch_strategy, 7: enable_migrate, 8: enable_defrag, 9: migrate_threshold
# 10: log_dirname


# We test 7 qps values rather than 5 in paper for each trace to show more stable results in once run.

# Estimated Time of this script: 47h
# sharegpt: 4h, burstgpt: 4h, 128-128: 2.5h, 256-256: 7h, 512-512: 14h, 128-512: 11.5h, 512-128: 3.5h


# sharegpt dataset
# 3.5 3.75 4.0 4.25 4.5
# 5x3=15 commands, 15min per command, 4h in total

# bash ./llumnix_exp_dataset 'first' ./config/serving_exp_dataset 3.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix'

# bash ./llumnix_exp_dataset 'first' ./config/serving_exp_dataset 3.75 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix'

# bash ./llumnix_exp_dataset 'first' ./config/serving_exp_dataset 4.0 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix'

# bash ./llumnix_exp_dataset 'first' ./config/serving_exp_dataset 4.25 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix'

# bash ./llumnix_exp_dataset 'first' ./config/serving_exp_dataset 4.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix'





bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 3.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'

bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 3.75 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'

bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 4.0 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'

bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 4.25 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'

bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 4.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'

exit 0

bash ./llumnix_exp_dataset 'all-wait' ./config/serving_exp_dataset 3.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all-wait'

bash ./llumnix_exp_dataset 'all-wait' ./config/serving_exp_dataset 3.75 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all-wait'

bash ./llumnix_exp_dataset 'all-wait' ./config/serving_exp_dataset 4.0 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all-wait'

bash ./llumnix_exp_dataset 'all-wait' ./config/serving_exp_dataset 4.25 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all-wait'

bash ./llumnix_exp_dataset 'all-wait' ./config/serving_exp_dataset 4.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all-wait'



exit 0
