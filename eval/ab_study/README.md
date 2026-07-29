# osunator A/B study harness

Blinded **two-alternative forced-choice (2AFC)** study: raters watch a pair of
replay clips — one human, one generated, same map and same section — and pick
the one they believe is human. Backend: FastAPI + SQLite. Frontend: static page.

Not a Turing test (no interrogation, no interaction): a discrimination task.

---

## Methodology

**Design.** 10 pairs per rater, one pair per held-out test map. ~30 s clips,
rendered with danser-go, no HUD (score/combo/accuracy counters would leak the
D-rank placeholder in generated replays). Both clips of a pair cover the same
time window of the same map, at the same resolution, framerate and encode
settings. Generation at temperature 0 (deterministic), same checkpoint as the
objective battery.

**Map and section selection.** The ten test maps were chosen by the author, who
was already familiar with the model's failure modes at the time of selection.
The set was constructed to span a range: two to three maps where the generated
replay was expected to be relatively easy to identify, two to three where the
author judged the generated replay to look *more* human than the human
reference, and the remainder drawn at random from the held-out test split.
Within each map, the ~30 s section was selected at random, and both clips of a
pair cover the same window.

Absolute detection rates therefore describe this stimulus set rather than an
unbiased sample of beatmaps, and the per-map spread is partly by construction.
Comparisons between rater groups are unaffected, since every rater judged the
same ten pairs.

**Slot assignment.** Which clip became slot A vs slot B was decided by a coin
flip per pair, executed mechanically by `make_pairs.py` (the flip, the filename
and the `a_is_human` flag are produced by the same line of code, so they cannot
disagree). Slot letters carry no meaning to raters: presentation order is
shuffled per rater, and which slot renders on the left is shuffled per pair per
rater. Clip filenames are content-free (`pairNN_A.mp4`).

**Raters.** Recruited via r/osugame. Self-reported experience bracket collected
before the task: `none` / `casual` / `5-6digit` / `1-4digit` / `lapsed`.
No accounts, no names. Cohort tags: `open` (public link), `friend` /
`reddit` (invite tokens), `test` (author).

**Exclusions** — documented, not deleted:
- One session (4/10, lapsed) was completed over screen-share with the author
  present and watching. Excluded from all analysis: the channel was unblinded,
  and stream compression degraded the very cursor-motion detail the task
  measures. That row lives in the local pre-deployment database, not the
  canonical one.
- Author test sessions were run against the deployed instance before any real
  rater participated, and were removed at that point. From launch onward no rows
  were deleted; unwanted rows are excluded by documented rule. The sole
  exception would be deletion at a rater's own request, which would be recorded
  here.
- Sessions started but never submitted (76) are excluded automatically by
  `analysis.py`, which prints the count.

**Collection window.** Opened 2026-07-22, closed 2026-07-29. The close was
extended from an earlier target of 2026-07-25 in order to recruit raters in the
`none` and `casual` brackets, which were badly underpowered at that point;
recruitment during the extension targeted non-players specifically.

Interim analyses were run on 2026-07-22 (n=11), 2026-07-23 (n=43) and
2026-07-25 (n=164). No design change followed from any of them; the extension
above was decided on bracket sample sizes, not on accuracy results.

**Known limitations of the protocol.**
- Open mode (`AB_ALLOW_OPEN=1`) was used for public recruitment: open-cohort
  participation is deduplicated heuristically (per-browser localStorage flag,
  per-IP-hash session cap, post-hoc inspection of hashed IPs), not enforced.
  Invite-token cohorts enforce single participation server-side.
- 10 pairs per rater bounds per-rater precision; the pooled 95% CI is ±2.7
  points at N=199, but small experience brackets remain wide (the casual
  bracket, n=8, spans [39.7, 70.3]).
- Experience brackets are self-reported and unverified.
- Maps were author-selected and deliberately stratified by expected
  detectability; absolute rates are specific to this stimulus set.

---

## Results

Collection ran 2026-07-22 to 2026-07-29. **199 raters submitted complete
sessions**, giving 1,990 judgments over 10 pairs. A further 76 raters started a
session and never submitted; these are excluded automatically and the count is
printed by `analysis.py`.

Confidence intervals below are **clustered by rater**: each rater contributes ten
correlated judgments, so treating the 1,990 as independent would overstate
precision. The clustered estimate is the mean of per-rater accuracies with the
standard error taken from the spread across raters. Per-map figures use exact
binomial tests instead, which is valid there because each rater contributes
exactly one judgment per map.

### By experience

| Bracket | Raters | Accuracy | 95% CI (rater-clustered) | z |
|---|---|---|---|---|
| Ranked, 1–4 digit | 69 | 69.3% | [64.5, 74.0] | 7.94 |
| Lapsed | 18 | 67.2% | [58.3, 76.2] | 3.78 |
| Ranked, 5–6 digit | 87 | 61.3% | [57.5, 65.0] | 5.93 |
| Casual | 8 | 55.0% | [39.7, 70.3] | 0.64 |
| Never played | 17 | 48.2% | [40.9, 55.6] | −0.47 |
| **Pooled** | **199** | **63.2%** | **[60.5, 65.9]** | **9.53** |

Raters who have never played perform at chance. Every bracket with real exposure
to the game performs well above it, and performance tracks familiarity rather
than current skill: lapsed players, who retain the perceptual knowledge but not
the mechanics, sit between the two active ranked brackets. The casual bracket
(n=8) is too small to support a claim in either direction.

### By map

| Map                | Accuracy | 95% CI | p |
|--------------------|---|---|---|
| words i never said | 73.9% | [67.4, 79.5] | 1.1e-11 |
| packet hero        | 70.9% | [64.2, 76.7] | 3.7e-09 |
| sakura no uta      | 69.8% | [63.1, 75.8] | 2.1e-08 |
| weenywalker        | 64.8% | [58.0, 71.1] | 3.5e-05 |
| no title           | 64.3% | [57.5, 70.6] | 6.5e-05 |
| zesty burrito      | 63.8% | [56.9, 70.2] | 1.0e-04 |
| amaoto noise       | 62.3% | [55.4, 68.8] | 6.0e-04 |
| replica            | 62.3% | [55.4, 68.8] | 6.0e-04 |
| super nuko world   | 51.3% | [44.4, 58.1] | 0.78 |
| my love            | 48.7% | [41.9, 55.6] | 0.78 |

Eight of ten maps were identified above chance; two were indistinguishable from
a coin flip. This spread is partly by construction — the stimulus set was chosen
to span expected difficulty (see *Map and section selection*) — so it is not an
unbiased estimate of how detectability varies across beatmaps. What it does show
is that the author's prior judgement predicted rater behaviour: the maps
selected as the model's best cases were the ones raters could not identify.
Characterizing what those clips have in common is left to future work.

### Distribution and diagnostics

Per-rater scores form a broad unimodal distribution centred around 6/10 with a
long right tail: 31 raters scored 9/10 or better, and 10 scored a perfect 10/10.
An earlier reading of these data as bimodal, based on the first ~40 raters, did
not survive the full sample.

Revised answers (34 of 1,990) were 58.8% accurate versus 63.3% for unrevised
ones; the difference is not significant and revision had no material effect.

Median per-rater decision time was 20.7 s, with the fastest decile under 1.6 s.
No attention-based exclusion was pre-declared, and none was applied: all 199
submitted raters are included in the figures above. `sensitivity.py` re-runs the
analysis under decision-time cutoffs for anyone who wants to check robustness.

Full output: `results-final.txt`.

---

## Privacy

Stored per rater: a random UUID, cohort, self-reported experience bracket,
timestamps, truncated user-agent, and a **salted SHA-256 hash of the IP
address** — the raw IP is never written. The salt lives only in the deployment
environment (`AB_IP_SALT`); without it the hashes are not reversible. No names,
emails or accounts.

Questions, or want your responses removed? Send me a Discord message: @aminewallah

The public data release (after collection closes) drops the `ip_hash`,
`user_agent` and `token` columns and reports small cohorts in aggregate only.

---

## Security invariant

The client NEVER receives ground-truth labels. Truth lives only in `pairs.json`
(server-side) and in the `slot_a_is_human` snapshot on each response row.
Scoring happens server-side; the reveal is served only after submission is
locked. `smoke_test.py` greps the session payload for label-shaped strings —
that check must pass before any deploy.

---

## Run locally

    uv sync                                 # deps incl. the `eval` group
    uv run python smoke_test.py             # must print ALL PASS
    AB_ALLOW_OPEN=1 AB_DB=scratch.db uv run uvicorn app:app --reload

`smoke_test.py` currently expects a real `pairs.json` / `tokens.json`

## Deployment (Railway)

- Service root directory: `eval/ab_study`; auto-deploys on push to master.
- Start command:
  `mkdir -p /data/clips && ln -sfn /data/clips static/clips && uvicorn app:app --host 0.0.0.0 --port $PORT`
- Volume mounted at `/data` holds `study.db`, `pairs.json`, `tokens.json`,
  `clips/` — these are deliberately absent from git and survive every deploy.
- Environment: `AB_DB`, `AB_PAIRS`, `AB_TOKENS` → `/data/...`;
  `AB_IP_SALT` (random hex); `AB_ALLOW_OPEN` (1 = public link, 0 = invite-only);
  `AB_SESSION_CAP` (default 10 tokenless sessions per IP-hash per day; invite
  tokens are exempt).
- `StaticFiles(..., follow_symlink=True)` is required: Starlette refuses to
  follow symlinks out of the static root by default, which 404s every clip.
- The canonical database is the deployed `/data/study.db`. The local
  `study.db` contains only the excluded pre-deployment session above.

Pull the database for analysis:

    curl -L "https://study.<domain>/admin/export?token=$AB_EXPORT_TOKEN" -o pulled_study.db
    uv run python analysis.py pulled_study.db pairs.json

## Analysis

    python analysis.py study.db pairs.json

Stdlib only. Reports rater-clustered accuracy with 95% confidence intervals and
z-scores (pooled, by experience bracket, by cohort), per-map accuracy with exact
two-sided binomial tests and Wilson intervals, the revision effect, the
per-rater score distribution, and decision-time deciles. Non-submitted raters
are excluded and the count printed.

The binomial test is computed in log space (via `lgamma`), since the direct
factorial form overflows above roughly n = 1000.

    python sensitivity.py study.db pairs.json [thresholds_ms...]

Post-hoc robustness check: re-runs the clustered analysis excluding raters whose
median decision time falls below a cutoff. No exclusion was applied to the
reported results.