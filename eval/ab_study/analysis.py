import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB = sys.argv[1] if len(sys.argv) > 1 else "pulled_study.db"
PAIRS_PATH = sys.argv[2] if len(sys.argv) > 2 else "pairs.json"

Z = 1.959963985


# ---------------------------------------------------------------- statistics

def log_binom_pmf(n, k, p):
    """log P(X = k) for X ~ Binomial(n, p). Computed in log space because
    math.comb(1990, 995) is a ~600-digit integer and overflows float."""
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial test: total probability of outcomes at most
    as likely as the observed one. May underflow to 0.0 for extreme results."""
    if n == 0:
        return 1.0
    logpk = log_binom_pmf(n, k, p)
    tol = 1e-7
    total = 0.0
    for i in range(n + 1):
        lp = log_binom_pmf(n, i, p)
        if lp <= logpk + tol:
            total += math.exp(lp)
    return min(1.0, total)


def wilson_ci(k, n):
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + Z**2 / n
    centre = phat + Z**2 / (2 * n)
    margin = Z * math.sqrt(phat * (1 - phat) / n + Z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def cluster_stats(rater_scores):
    """Rater-clustered statistics from [(correct, n_answered), ...].

    Raters answer the same number of pairs, so the cluster-robust estimate is
    the mean of per-rater accuracies, with the standard error taken from the
    spread ACROSS raters rather than across individual judgments.
    """
    m = len(rater_scores)
    total_k = sum(k for k, _ in rater_scores)
    total_n = sum(n for _, n in rater_scores)
    nan = float("nan")
    if m == 0 or total_n == 0:
        return {"m": m, "k": total_k, "n": total_n, "phat": nan,
                "lo": nan, "hi": nan, "z": nan}

    props = [k / n for k, n in rater_scores if n]
    phat = sum(props) / len(props)
    if m > 1:
        var = sum((p - phat) ** 2 for p in props) / (m - 1)
        se = math.sqrt(var / m)
    else:
        se = nan
    z = (phat - 0.5) / se if se and se > 0 else nan
    return {"m": m, "k": total_k, "n": total_n, "phat": phat,
            "lo": phat - Z * se, "hi": phat + Z * se, "z": z}


def fmt_p(p):
    if p == 0.0:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cluster_line(label, scores):
    s = cluster_stats(scores)
    if s["m"] == 0:
        return f"{label:<26} (no raters)"
    if s["m"] == 1:
        return (f"{label:<26} {s['k']:>5}/{s['n']:<5} = {s['k']/s['n']:6.1%}"
                f"   (1 rater - no CI)")
    return (f"{label:<26} {s['k']:>5}/{s['n']:<5} = {s['phat']:6.1%}"
            f"   95% CI [{s['lo']:5.1%}, {s['hi']:5.1%}]"
            f"   z = {s['z']:5.2f}   ({s['m']} raters)")


def binom_line(label, k, n):
    if n == 0:
        return f"{label:<26} n=0"
    lo, hi = wilson_ci(k, n)
    return (f"{label:<26} {k:>5}/{n:<5} = {k/n:6.1%}"
            f"   95% CI [{lo:5.1%}, {hi:5.1%}]"
            f"   p = {fmt_p(binom_two_sided_p(k, n))}")


# ---------------------------------------------------------------- main

def main():
    pairs = json.loads(Path(PAIRS_PATH).read_text())
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    submitted = {r["id"]: r for r in conn.execute(
        "SELECT * FROM raters WHERE submitted_at IS NOT NULL")}
    unsubmitted = conn.execute(
        "SELECT COUNT(*) c FROM raters WHERE submitted_at IS NULL").fetchone()["c"]

    rows = [r for r in conn.execute("SELECT * FROM responses")
            if r["rater_id"] in submitted]

    print(f"raters:    {len(submitted)} submitted, {unsubmitted} excluded (never submitted)")
    print(f"judgments: {len(rows)} from submitted raters")
    print(f"pairs:     {len(pairs)}\n")

    def correct(r):
        return (r["chosen_slot"] == "A") == bool(r["slot_a_is_human"])

    per_rater = defaultdict(lambda: [0, 0])
    for r in rows:
        per_rater[r["rater_id"]][0] += correct(r)
        per_rater[r["rater_id"]][1] += 1

    incomplete = {rid[:8]: n for rid, (_, n) in per_rater.items() if n != len(pairs)}
    if incomplete:
        print(f"NOTE: {len(incomplete)} submitted rater(s) with != {len(pairs)} "
              f"answers: {incomplete}\n")

    print("=" * 78)
    print("RATER-CLUSTERED (headline numbers - use these in the paper)")
    print("=" * 78)
    all_scores = [tuple(v) for v in per_rater.values()]
    print(cluster_line("POOLED", all_scores))

    for field in ("experience", "cohort"):
        print(f"\nby {field}:")
        groups = defaultdict(list)
        for rid, (k, n) in per_rater.items():
            groups[submitted[rid][field]].append((k, n))
        for g in sorted(groups, key=lambda g: -len(groups[g])):
            print("  " + cluster_line(str(g), groups[g]))

    print("\n" + "=" * 78)
    print("PER-MAP (exact binomial valid: one judgment per rater per map)")
    print("=" * 78)
    by_map = defaultdict(lambda: [0, 0])
    for r in rows:
        m = pairs.get(r["pair_id"], {}).get("map", r["pair_id"])
        by_map[m][0] += correct(r)
        by_map[m][1] += 1
    for m in sorted(by_map, key=lambda m: -by_map[m][0] / max(1, by_map[m][1])):
        print("  " + binom_line(str(m)[:25], *by_map[m]))

    print("\n" + "=" * 78)
    print("DIAGNOSTICS")
    print("=" * 78)
    print("\nrevision effect:")
    for flag, label in ((0, "final = first answer"), (1, "revised answers")):
        sub = [r for r in rows if r["revised"] == flag]
        print("  " + binom_line(label, sum(correct(r) for r in sub), len(sub)))

    print("\nper-rater score distribution:")
    hist = defaultdict(int)
    for k, n in per_rater.values():
        hist[k] += 1
    top = max(hist.values()) if hist else 0
    for score in range(0, len(pairs) + 1):
        c = hist.get(score, 0)
        bar = "#" * round(40 * c / top) if top else ""
        print(f"  {score:2d}/{len(pairs)}  {c:4d}  {bar}")

    print("\ndecision time (median ms per rater, deciles):")
    times = defaultdict(list)
    for r in rows:
        if r["decision_ms"] is not None:
            times[r["rater_id"]].append(r["decision_ms"])
    medians = sorted(sorted(v)[len(v) // 2] for v in times.values() if v)
    if medians:
        for q in range(0, 11):
            i = min(len(medians) - 1, q * (len(medians) - 1) // 10)
            print(f"  p{q*10:<3} {medians[i]:>8} ms")


if __name__ == "__main__":
    main()