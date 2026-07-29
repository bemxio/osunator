"""
Robustness check: does the result survive excluding fast (likely inattentive)
raters? Post-hoc sensitivity analysis — report alongside the main result and
label it as post-hoc, do not silently substitute it.

Usage:
    python sensitivity.py study.db pairs.json [thresholds_ms...]
    (default thresholds: 0 2000 5000 8000 — median decision time per rater)
"""
import json, math, sqlite3, sys
from collections import defaultdict
from pathlib import Path

DB = sys.argv[1] if len(sys.argv) > 1 else "study.db"
PAIRS = sys.argv[2] if len(sys.argv) > 2 else "pairs.json"
THRESHOLDS = [int(x) for x in sys.argv[3:]] or [0, 2000, 5000, 8000]
Z = 1.959963985


def cluster(scores):
    m = len(scores)
    if m < 2:
        return None
    props = [k / n for k, n in scores]
    p = sum(props) / m
    var = sum((x - p) ** 2 for x in props) / (m - 1)
    se = math.sqrt(var / m)
    return p, p - Z * se, p + Z * se, ((p - 0.5) / se if se else float("nan")), m


def main():
    pairs = json.loads(Path(PAIRS).read_text())
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    subs = {r["id"]: r for r in conn.execute(
        "SELECT * FROM raters WHERE submitted_at IS NOT NULL")}
    rows = [r for r in conn.execute("SELECT * FROM responses")
            if r["rater_id"] in subs]

    def ok(r):
        return (r["chosen_slot"] == "A") == bool(r["slot_a_is_human"])

    tally, times = defaultdict(lambda: [0, 0]), defaultdict(list)
    for r in rows:
        tally[r["rater_id"]][0] += ok(r)
        tally[r["rater_id"]][1] += 1
        if r["decision_ms"] is not None:
            times[r["rater_id"]].append(r["decision_ms"])
    med = {rid: sorted(v)[len(v) // 2] for rid, v in times.items() if v}

    print("median decision time by experience bracket:")
    by_exp = defaultdict(list)
    for rid, t in med.items():
        by_exp[subs[rid]["experience"]].append(t)
    for e in sorted(by_exp, key=lambda e: -len(by_exp[e])):
        v = sorted(by_exp[e])
        fast = sum(1 for x in v if x < 5000)
        print(f"  {e:<12} n={len(v):3d}  median={v[len(v)//2]:>7} ms"
              f"  <5s: {fast} ({fast/len(v):.0%})")

    for thr in THRESHOLDS:
        keep = {rid for rid in tally if med.get(rid, 10**9) >= thr}
        print(f"\n--- excluding raters with median decision time < {thr} ms "
              f"({len(tally)-len(keep)} of {len(tally)} raters dropped) ---")
        groups = defaultdict(list)
        allsc = []
        for rid in keep:
            sc = tuple(tally[rid])
            allsc.append(sc)
            groups[subs[rid]["experience"]].append(sc)
        for label, sc in [("POOLED", allsc)] + sorted(
                groups.items(), key=lambda kv: -len(kv[1])):
            c = cluster(sc)
            if c is None:
                print(f"  {label:<12} (n<2)")
                continue
            p, lo, hi, z, m = c
            print(f"  {label:<12} {p:6.1%}  95% CI [{lo:5.1%}, {hi:5.1%}]"
                  f"  z={z:6.2f}  ({m} raters)")


if __name__ == "__main__":
    main()