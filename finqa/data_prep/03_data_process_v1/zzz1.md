(openr1) root@autodl-container-8eda4eaad0-8e4d3743:~/autodl-tmp/open-r1/03_data_process_v1# cat \
/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1/reports/preprocess_report.json
{
  "pipeline_version": "v1.1-lowmem",
  "config": {
    "data_dir": "/root/autodl-tmp/project/finqa_posttrain/data/raw/FinQA/dataset",
    "output_dir": "/root/autodl-tmp/project/finqa_posttrain/data/processed_v1_1_1",
    "model_path": "/root/autodl-tmp/models/Qwen3-0.6B-Base",
    "max_seq_length": 2048,
    "sft_response_reserve": 384,
    "grpo_prompt_max_tokens": 1792,
    "system_prompt": "You are a financial numerical reasoning assistant. Use the financial report and table to solve the question. Show concise calculation steps and place the final answer inside <answer>...</answer>.",
    "numeric_rel_tol": 0.001,
    "numeric_abs_tol": 0.01,
    "percent_abs_tol": 0.1,
    "keep_suspicious": true
  },
  "overlap_check": {
    "id_overlap_train_dev": 0,
    "filename_overlap_train_dev": 0,
    "id_overlap_train_test": 0,
    "filename_overlap_train_test": 0,
    "id_overlap_dev_test": 0,
    "filename_overlap_dev_test": 0
  },
  "canonical_summary": {
    "train": {
      "num_examples": 6251,
      "annotation_quality": {
        "high": 5864,
        "medium": 387
      },
      "annotation_flags": {
        "answer_step_mismatch": 387
      },
      "semantic_status": {
        "not_flagged": 6250,
        "suspicious": 1
      },
      "semantic_flags": {
        "question_asks_max_but_ops_use_min": 1
      },
      "difficulty": {
        "medium": 2013,
        "hard": 521,
        "easy": 3717
      },
      "gold_source": {
        "answer": 6203,
        "steps[-1].res": 48
      },
      "normalized_operator_counts": {
        "divide": 4445,
        "multiply": 567,
        "greater": 124,
        "subtract": 2739,
        "add": 1512,
        "average": 95,
        "max": 48,
        "min": 27,
        "sum": 36,
        "exp": 5
      }
    },
    "dev": {
      "num_examples": 883,
      "annotation_quality": {
        "high": 844,
        "medium": 39
      },
      "annotation_flags": {
        "answer_step_mismatch": 39
      },
      "semantic_status": {
        "not_flagged": 883
      },
      "semantic_flags": {},
      "difficulty": {
        "easy": 523,
        "medium": 287,
        "hard": 73
      },
      "gold_source": {
        "answer": 871,
        "steps[-1].res": 12
      },
      "normalized_operator_counts": {
        "divide": 636,
        "subtract": 416,
        "multiply": 83,
        "add": 180,
        "average": 19,
        "sum": 4,
        "max": 8,
        "greater": 10,
        "min": 5,
        "exp": 1
      }
    },
    "test": {
      "num_examples": 1147,
      "annotation_quality": {
        "high": 1091,
        "medium": 56
      },
      "annotation_flags": {
        "answer_step_mismatch": 56
      },
      "semantic_status": {
        "not_flagged": 1145,
        "suspicious": 2
      },
      "semantic_flags": {
        "comparison_yesno_question_without_greater_op": 2
      },
      "difficulty": {
        "easy": 654,
        "medium": 409,
        "hard": 84
      },
      "gold_source": {
        "answer": 1133,
        "steps[-1].res": 14
      },
      "normalized_operator_counts": {
        "subtract": 521,
        "divide": 820,
        "greater": 20,
        "average": 15,
        "add": 260,
        "max": 10,
        "multiply": 109,
        "sum": 10,
        "exp": 3,
        "min": 4
      }
    }
  },
  "truncation_summary": {
    "sft_train": {
      "num_examples": 6251,
      "num_truncated": 251,
      "truncation_rate": 0.040153575427931534,
      "mean_original_prompt_tokens": 1067.020156774916,
      "mean_final_prompt_tokens": 1040.9235322348425,
      "max_original_prompt_tokens": 3300,
      "max_final_prompt_tokens": 1661
    },
    "sft_dev": {
      "num_examples": 883,
      "num_truncated": 40,
      "truncation_rate": 0.045300113250283124,
      "mean_original_prompt_tokens": 1056.748584371461,
      "mean_final_prompt_tokens": 1035.6930917327293,
      "max_original_prompt_tokens": 2163,
      "max_final_prompt_tokens": 1660
    },
    "grpo_train": {
      "num_examples": 6251,
      "num_truncated": 185,
      "truncation_rate": 0.029595264757638778,
      "mean_original_prompt_tokens": 1067.020156774916,
      "mean_final_prompt_tokens": 1048.3183490641497,
      "max_original_prompt_tokens": 3300,
      "max_final_prompt_tokens": 1792
    },
    "eval_dev": {
      "num_examples": 883,
      "num_truncated": 36,
      "truncation_rate": 0.04077010192525481,
      "mean_original_prompt_tokens": 1056.748584371461,
      "mean_final_prompt_tokens": 1042.4915062287655,
      "max_original_prompt_tokens": 2163,
      "max_final_prompt_tokens": 1788
    },
    "eval_test": {
      "num_examples": 1147,
      "num_truncated": 29,
      "truncation_rate": 0.025283347863993024,
      "mean_original_prompt_tokens": 1054.7000871839582,
      "mean_final_prompt_tokens": 1038.9380993897123,
      "max_original_prompt_tokens": 2541,
      "max_final_prompt_tokens": 1788
    }
  },
  "notes": [
    "Low-memory version: one split at a time, incremental JSONL writes.",
    "SFT train and GRPO train intentionally share FinQA train.",
    "Dev/test are never used for optimization.",
    "gold_inds/retrieved evidence are never inserted into prompts.",
    "semantic_status=not_flagged does NOT mean human-verified correctness.",
    "Baseline keeps suspicious samples; filtering is reserved for ablation."
  ]