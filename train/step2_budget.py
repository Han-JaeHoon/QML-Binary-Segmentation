"""
Step 2 — pick a COMMON optimization budget for the centre-branch M1/M2 runs.

PRE-REGISTERED: this criterion is written and committed BEFORE the diagnostic
runs are executed, so the budget cannot be chosen post hoc to favour a model.

Goal (deliberately narrow):
    find a budget at which optimization has effectively converged,
    NOT the budget that maximizes dev validation performance.
Selecting on dev validation AP would leak: the dev val cities reappear as outer
validation cities in the 5-fold, so tuning on them biases the later estimate.
Cheap-val AP is therefore logged as a diagnostic only and is NOT used here.

CRITERION
    L_e            = mean train BCE of epoch e
    MA5_e          = mean(L_{e-4..e})                      (5-epoch moving average)
    plateau at e   iff |MA5_e - MA5_{e-5}| < DELTA for at least SUSTAIN
                   consecutive epochs
    DELTA   = 0.002      (absolute BCE)
    SUSTAIN = 10         (epochs)
    budget  = ceil((first plateau epoch of the SLOWER model + margin) / 10) * 10
    MARGIN  = 5 epochs; if no model plateaus, keep the full diagnostic length.

The budget is COMMON to M1 and M2 — never give the two models different budgets;
take the slower-converging one and round up.

REFERENCE: the centre branch trains at a ~24.8 % positive prior, so a constant
predictor scores BCE = H(0.248) ~= 0.562. A run that does not fall clearly below
this has not learned anything.
"""
import json, os, sys
import numpy as np

DELTA = 0.002
SUSTAIN = 10
MARGIN = 5
PI_TRAIN = 0.248
BCE_CONST = float(-(PI_TRAIN * np.log(PI_TRAIN) + (1 - PI_TRAIN) * np.log(1 - PI_TRAIN)))


def moving_avg(x, k=5):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - k + 1):i + 1].mean() for i in range(len(x))])


def plateau_epoch(bce, delta=DELTA, sustain=SUSTAIN, k=5):
    """First epoch (1-indexed) from which |MA5_e - MA5_{e-5}| < delta holds for
    `sustain` consecutive epochs. None if never."""
    ma = moving_avg(bce, k)
    ok = np.zeros(len(ma), dtype=bool)
    for e in range(k, len(ma)):
        ok[e] = abs(ma[e] - ma[e - k]) < delta
    run = 0
    for e in range(len(ok)):
        run = run + 1 if ok[e] else 0
        if run >= sustain:
            return e - sustain + 2          # 1-indexed epoch where the run began
    return None


def load(path):
    rows = [json.loads(l) for l in open(path)]
    ep = [r for r in rows if "epoch" in r]
    return {
        "bce": [r["train_BCE"] for r in ep],
        "ap": [r["cheap_AP"] for r in ep],
        "gn": [r["grad_norm"] for r in ep],
        "wt": [r["wall_time"] for r in ep],
        "cks": [r.get("stream_checksum") for r in ep],
    }


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "../results/runs/step2_m1.jsonl",
        "../results/runs/step2_m2.jsonl",
    ]
    runs = {os.path.basename(p).replace(".jsonl", ""): load(p) for p in paths}

    print(f"constant-predictor reference: BCE = H({PI_TRAIN}) = {BCE_CONST:.4f}")
    print(f"pre-registered criterion: |MA5_e - MA5_(e-5)| < {DELTA} sustained "
          f"{SUSTAIN} epochs; common budget = slower plateau + {MARGIN}, rounded up to 10\n")

    # paired-stream verification
    names = list(runs)
    if len(names) == 2:
        a, b = runs[names[0]]["cks"], runs[names[1]]["cks"]
        n = min(len(a), len(b))
        same = all(x == y for x, y in zip(a[:n], b[:n])) and None not in a[:n]
        print(f"paired stream: epoch checksums identical over {n} epochs -> "
              f"{'YES' if same else 'NO (streams differ!)'}\n")

    plateaus = {}
    for name, r in runs.items():
        bce = r["bce"]; ma = moving_avg(bce)
        pe = plateau_epoch(bce); plateaus[name] = pe
        below = next((i + 1 for i, v in enumerate(bce) if v < BCE_CONST), None)
        print(f"{name}: {len(bce)} epochs | BCE {bce[0]:.4f} -> {bce[-1]:.4f} "
              f"(min {min(bce):.4f}, MA5 final {ma[-1]:.4f})")
        print(f"    below constant baseline from epoch {below} | "
              f"plateau epoch {pe if pe else 'not reached'} | "
              f"final cheap AP {r['ap'][-1]:.4f} (diagnostic only) | "
              f"{r['wt'][-1]/len(bce):.1f} s/epoch")

    vals = [v for v in plateaus.values() if v is not None]
    if len(vals) == len(runs) and vals:
        budget = int(np.ceil((max(vals) + MARGIN) / 10.0) * 10)
        print(f"\nslower model plateaus at epoch {max(vals)} -> "
              f"COMMON BUDGET = {budget} epochs")
    else:
        n = max(len(r["bce"]) for r in runs.values())
        print(f"\nat least one model did not plateau -> keep full diagnostic "
              f"length: COMMON BUDGET = {n} epochs")
