# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/monitor_eval.py
Command: python3 src/monitor_eval.py --n_eval 250 --seed 42
Git-Commit: a7623bab7b63a571a2b01842cdde878b45cd7d38
Seeds: 42 (MMLU and ARC eval sampling in src/task.py; greedy decoding so generation is deterministic; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu, in-domain = the GRPO training domain) and ARC-Challenge (allenai/ai2_arc, out-of-domain) eval slices via src/task.py; Qwen2.5-1.5B-Instruct base=checkpoint-0 and outcome-only GRPO RL=checkpoint-final trained on MMLU (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130
Analysis-Command: cd results/real && python3 recompute.py  (each section value = mean(val) with 95% bootstrap CI)
Columns:
  section (one of the 12 per-cell arms {base,rl}_{mmlu,arc}_{intact_flip,mismatch_flip,acc}); eval_order (position within the paired intersection); val (0/1: flip or correctness)
