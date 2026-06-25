"""recompute.py — reproduce analysis_summary.txt from per-example data alone (stdlib).
Each section's value = mean(val) over eval_points with seeded bootstrap 95% CI.
Gate: cd results/real && python3 recompute.py | diff - analysis_summary.txt  (empty)."""
import json, random
from collections import defaultdict
BOOT_N, BOOT_SEED = 2000, 42
SECTIONS = [
    ("Base, MMLU (in-domain): mismatch-CoT flip rate (monitorability)", "base_mmlu_mismatch_flip"),
    ("RL, MMLU (in-domain): mismatch-CoT flip rate (monitorability)", "rl_mmlu_mismatch_flip"),
    ("Base, ARC (out-of-domain): mismatch-CoT flip rate (monitorability)", "base_arc_mismatch_flip"),
    ("RL, ARC (out-of-domain): mismatch-CoT flip rate (monitorability)", "rl_arc_mismatch_flip"),
    ("Base, MMLU: intact-CoT flip rate (control)", "base_mmlu_intact_flip"),
    ("RL, MMLU: intact-CoT flip rate (control)", "rl_mmlu_intact_flip"),
    ("Base, ARC: intact-CoT flip rate (control)", "base_arc_intact_flip"),
    ("RL, ARC: intact-CoT flip rate (control)", "rl_arc_intact_flip"),
    ("Base, MMLU: accuracy", "base_mmlu_acc"),
    ("RL, MMLU: accuracy", "rl_mmlu_acc"),
    ("Base, ARC: accuracy", "base_arc_acc"),
    ("RL, ARC: accuracy", "rl_arc_acc"),
]
def pct(s,q):
    if not s: return float("nan")
    p=q/100.0*(len(s)-1); lo=int(p); f=p-lo
    return s[lo]*(1-f)+s[lo+1]*f if lo+1<len(s) else s[lo]
def rate_ci(vals):
    n=len(vals); point=sum(vals)/n if n else float("nan")
    rng=random.Random(BOOT_SEED); boots=[]
    for _ in range(BOOT_N):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    boots.sort(); return point, pct(boots,2.5), pct(boots,97.5), n
def main():
    by=defaultdict(list)
    for line in open("eval_points.jsonl"):
        line=line.strip()
        if line: r=json.loads(line); by[r["section"]].append((r["eval_order"], r["val"]))
    for k in by: by[k].sort()
    L=["# CoT-monitorability decay from domain-specific RL: local or generalized?",
       "","Model: Qwen2.5-1.5B-Instruct, base (checkpoint-0) vs outcome-only GRPO trained on MMLU (checkpoint-final).",
       "Monitorability proxy: mismatch-CoT flip rate (answer changes when the model's own CoT is swapped for another's).",
       "Higher = answer depends on CoT = more monitorable. In-domain = MMLU, out-of-domain = ARC-Challenge.",
       "Paired on items both models retain per domain. Bootstrap 2000, seed 42.",""]
    for title,key in SECTIONS:
        vals=[v for _,v in by.get(key,[])]; p,lo,hi,n=rate_ci(vals)
        L.append("## %s" % title); L.append("  value = %.4f  (95%% CI %.4f-%.4f, n=%d)" % (p,lo,hi,n)); L.append("")
    print("\n".join(L).rstrip("\n"))
if __name__=="__main__": main()
