import csv, json
from pathlib import Path
OUT = Path("results/real"); GIT = "a7623bab7b63a571a2b01842cdde878b45cd7d38"
rows = list(csv.DictReader(open(OUT/"curve.csv")))

def sidecar(name, columns, ac):
    (OUT/(name+".md")).write_text(
"""# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/monitor_eval.py
Command: python3 src/monitor_eval.py --n_eval 250 --seed 42
Git-Commit: %s
Seeds: 42 (MMLU and ARC eval sampling in src/task.py; greedy decoding so generation is deterministic; 2000-resample percentile bootstrap)
Source-Data: MMLU (cais/mmlu, in-domain = the GRPO training domain) and ARC-Challenge (allenai/ai2_arc, out-of-domain) eval slices via src/task.py; Qwen2.5-1.5B-Instruct base=checkpoint-0 and outcome-only GRPO RL=checkpoint-final trained on MMLU (reused from ~/projects/rl-selfcot-causal/runs/rl), RTX 5090, 2026-06-24, torch 2.12 cu130
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, ac, columns))

sidecar("curve.csv",
        "  metric (section name: {base,rl}_{mmlu,arc}_{intact,mismatch}_flip = self-CoT corruption flip rate, mismatch = monitorability proxy [higher = answer depends on CoT = monitorable], intact = control [prefill own full CoT]; {base,rl}_{mmlu,arc}_acc = MMLU/ARC accuracy; monitorability_decay_{mmlu,arc} = base_mismatch_flip minus rl_mismatch_flip per domain [positive = decay]; decay_indomain_minus_ood = decay_mmlu minus decay_arc);\n"
        "  value (rate/accuracy 0-1, or signed decay for the decay rows); n (paired item intersection per domain: 96 for MMLU, 158 for ARC)",
        "cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty); the 12 per-cell rates are reproduced from eval_points.jsonl, the decay rows are arithmetic on the mismatch rows")
sidecar("eval_points.jsonl",
        "  section (one of the 12 per-cell arms {base,rl}_{mmlu,arc}_{intact_flip,mismatch_flip,acc}); eval_order (position within the paired intersection); val (0/1: flip or correctness)",
        "cd results/real && python3 recompute.py  (each section value = mean(val) with 95% bootstrap CI)")

def w(stem, metrics, desc):
    rs=[r for r in rows if r["metric"] in metrics]
    with open(OUT/(stem+".csv"),"w",newline="") as f:
        wr=csv.DictWriter(f, fieldnames=["metric","value","n"]); wr.writeheader()
        for r in rs: wr.writerow(r)
    (OUT/(stem+".md")).write_text("# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/monitor_eval.py + src/meta.py. Git-Commit: %s\n"%(stem,stem,desc,GIT))

w("figure_main", {"base_mmlu_mismatch_flip","rl_mmlu_mismatch_flip","base_arc_mismatch_flip","rl_arc_mismatch_flip"}, "Mismatch-CoT flip rate (monitorability), base vs RL, MMLU (in-domain) vs ARC (OOD).")
w("figure_decay", {"monitorability_decay_mmlu","monitorability_decay_arc","decay_indomain_minus_ood"}, "Monitorability decay (base minus RL mismatch flip) per domain, and the in-domain-minus-OOD difference.")
w("figure_intact", {"base_mmlu_intact_flip","rl_mmlu_intact_flip","base_arc_intact_flip","rl_arc_intact_flip"}, "Intact-CoT flip rate (control), base vs RL, both domains.")
w("figure_accuracy", {"base_mmlu_acc","rl_mmlu_acc","base_arc_acc","rl_arc_acc"}, "Task accuracy, base vs RL, MMLU vs ARC.")

(OUT/"sources.json").write_text(json.dumps({"metrics":{"*":{"csv":"curve.csv"}},"per_example":["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")
