#!/usr/bin/env python3
"""Subject distribution of the demos DSPy BootstrapFewShot selected, per model.
The mechanism smoking gun: correctness-based selection should over-pick reasoning
subjects (math/physics), starving knowledge subjects (ona_tili/tarix).
Usage: demo_dist.py [results_dir] [dataset.json]"""
import json, glob, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else "results/main"
ds = sys.argv[2] if len(sys.argv) > 2 else "data/DTM_benchmark.json"

q2subj = {}
for row in json.load(open(ds)):
    q2subj[row["question"].strip()] = row.get("subject", "?")

# fraction of the *train pool* that is each subject, for the baseline expectation
pool = collections.Counter(q2subj.values())
tot = sum(pool.values())
print("Dataset subject mix: " + "  ".join(f"{s}={100*n/tot:.0f}%" for s, n in pool.most_common()))
print()
print(f"{'model':14}{'#demos':>7}   demo subject distribution")
print("-" * 62)
agg = collections.Counter()
for f in sorted(glob.glob(f"{d}/*_bootstrap.json")):
    prog = json.load(open(f))
    demos = prog.get("predict", {}).get("demos", [])
    subs = collections.Counter()
    for dm in demos:
        q = (dm.get("question") or "").strip()
        subs[q2subj.get(q, "UNMATCHED")] += 1
    agg.update(subs)
    model = f.split("/")[-1].replace("_bootstrap.json", "")
    dist = "  ".join(f"{s}:{n}" for s, n in subs.most_common())
    print(f"{model:14}{len(demos):>7}   {dist}")

print("-" * 62)
tot_d = sum(agg.values())
print(f"{'ALL':14}{tot_d:>7}   " + "  ".join(f"{s}:{n}({100*n/tot_d:.0f}%)" for s, n in agg.most_common()))
print("\nReasoning (math+physics) share of demos vs dataset:")
rz = agg.get("matematika", 0) + agg.get("fizika", 0)
rz_ds = pool.get("matematika", 0) + pool.get("fizika", 0)
print(f"  demos: {100*rz/tot_d:.0f}%   dataset: {100*rz_ds/tot:.0f}%   -> over-selection factor {(rz/tot_d)/(rz_ds/tot):.1f}x")
