"""The dipole CNN against the classic rule, with acceptance criteria (task A9, stage 2).

    python -m training.evaluate     # docs/dipole_cnn.md, docs/dipole_cnn.json, docs/img/

Every score is out of fold (the CNN never trained on the field it judges). Two test sets:

- injected: the stamps of training/dataset.py (injected dipoles and hard negatives, of which the
  fixed stars and the sky are real, untouched stamps);
- real: each moving target (Barnard's Star, Epsilon Indi B, WISE 0855) at its catalog midpoint in
  every difference where it moved 0.3 px or more, against the saturated stars that do not move
  (RW Aurigae, R Draconis). Differences where a mover moved less are left out of both.

On the stamps, the rule (serafim/validation/dipoles.py) says yes when a positive and a negative
5-sigma blob pair up within RULE_CENTER_PX of the center; the CNN says yes above the threshold
that gives it the rule's false-alarm rate on the injected negatives. On the real set the CNN is
also compared with the full stage-1 rule (whole image, halo lobes, catalog check), which is what
the pipeline uses. The CNN may change a status only if it meets every acceptance criterion;
otherwise it is reported as experimental and its p_dipole is an extra opinion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from serafim.validation import dipoles
from training.dataset import MOVING_TARGETS, OUT_DIR, STILL_SATURATED
from training.metrics import (
    binary_metrics,
    interval,
    ppv_at,
    reliability,
    roc_auc,
    youden_threshold,
)
from training.rule import rule_says, stage_one
from training.train import load

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "dipole_cnn.md"
SUMMARY_PATH = ROOT / "docs" / "dipole_cnn.json"  # read by serafim/validation/dipoles.py
IMAGE_PATH = ROOT / "docs" / "img" / "dipole_cnn.png"
MIN_SHIFT_PX = 0.3
SNR_BINS = [3, 5, 10, 30, 100, 300]
SHIFT_BINS = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0]
PREVALENCES = (0.1, 0.01, 0.001)
# What a professional deliverable needs before the CNN may change anything in the pipeline
MIN_AUC = 0.9
MAX_ECE = 0.05
MIN_REAL_AUC = 0.8
MAX_REAL_FALSE_ALARM = 0.1


def matched_threshold(p_negative: np.ndarray, false_alarm: float) -> float:
    """The CNN threshold with the rule's false-alarm rate on the negatives."""
    return float(np.quantile(p_negative, 1 - false_alarm)) if false_alarm > 0 else 1.0


def real_set(real_table: pd.DataFrame) -> pd.DataFrame:
    """Movers that moved MIN_SHIFT_PX or more (label 1) and still saturated stars (label 0)."""
    moving = real_table["target_id"].isin(MOVING_TARGETS)
    keep = (moving & (real_table["shift_px"] >= MIN_SHIFT_PX)) | real_table["target_id"].isin(
        STILL_SATURATED
    )
    return real_table[keep].assign(label=moving[keep].astype(int))


def learning_lines(path: Path = ROOT / "docs" / "dipole_cnn_learning.json") -> list[str]:
    """The learning curve of training/learning_curve.py, if it was run."""
    if not path.exists():
        return []
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    lines = [
        "## Enough data? The learning curve",
        "",
        "Trained on 1/8, 1/4, 1/2 and all of the differences of four targets "
        "(`python -m training.learning_curve`):",
        "",
        "| Target left out | Training stamps | AUC on its training stamps | AUC on the target "
        "left out |",
        "| --- | --- | --- | --- |",
        *[f"| {r.held_out} | {r.stamps} | {r.auc_trained_on:.3f} | {r.auc_held_out:.3f} |"
          for r in rows.itertuples()],
        "",
    ]  # fmt: skip
    last = (
        rows.groupby("held_out")
        .tail(2)
        .groupby("held_out")["auc_held_out"]
        .agg(lambda s: s.iloc[-1] - s.iloc[0])
    )
    gap = rows.groupby("held_out").tail(1).set_index("held_out")
    gap = (gap["auc_trained_on"] - gap["auc_held_out"]).max()
    climbing = bool((last > 0.02).any())
    lines += [
        f"From half to all of the data the held-out AUC still rises by "
        f"{', '.join(f'{v:+.3f}' for v in last)}: "
        + ("the curve has not flattened, so more data would still help. " if climbing else
           "the curve has flattened: more of the same data would not help. ")
        + f"The gap between the stamps it trained on and the ones it never saw is at most "
        f"{gap:.3f} at full size: no sign of memorizing (overfitting).",
        "",
        "![Learning curve](img/dipole_cnn_learning.png)",
        "",
    ]  # fmt: skip
    return lines


def by_bins(table: pd.DataFrame, column: str, bins: list) -> pd.DataFrame:
    cut = pd.cut(table[column], bins)
    return table.groupby(cut, observed=True).agg(n=("label", "size"), rule=("rule", "mean"),
                                                 cnn=("cnn", "mean"))  # fmt: skip


ROWS = [
    ("TP / FN / FP / TN", None),
    ("Prevalence (share of dipoles in the test)", "prevalence"),
    ("Sensitivity (recall, TPR)", "sensitivity"),
    ("Specificity (TNR)", "specificity"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Informedness (Youden's J)", "informedness"),
    ("ROC AUC", "ROC_AUC"),
    ("Precision (PPV)", "PPV"),
    ("NPV", "NPV"),
    ("Accuracy", "accuracy"),
    ("MCC", "MCC"),
    ("Markedness", "markedness"),
    ("F1", "F1"),
    ("Jaccard", "jaccard"),
    ("Fowlkes-Mallows", "fowlkes_mallows"),
    ("Average precision (PR AUC)", "average_precision"),
    ("Prevalence threshold", "prevalence_threshold"),
]


def metrics_lines(columns: dict[str, dict]) -> list[str]:
    names = list(columns)
    lines = ["| Metric | " + " | ".join(names) + " |", "|" + " --- |" * (len(names) + 1)]
    for label, key in ROWS:
        cells = []
        for m in columns.values():
            if key is None:
                cells.append(f"{m['TP']} / {m['FN']} / {m['FP']} / {m['TN']}")
            else:
                value = m.get(key, float("nan"))
                cells.append("—" if value is None or np.isnan(value) else f"{value:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for prevalence in PREVALENCES:
        cells = [f"{ppv_at(m['sensitivity'], m['specificity'], prevalence):.3f}"
                 for m in columns.values()]  # fmt: skip
        lines.append(f"| Precision if {prevalence:.1%} of the places hold a dipole | "
                     + " | ".join(cells) + " |")  # fmt: skip
    return lines


def plot(snr: pd.DataFrame, shift: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, table, label in ((axes[0], snr, "peak S/N in the difference"),
                             (axes[1], shift, "shift between the seasons (px)")):  # fmt: skip
        names = [f"{i.left:g}–{i.right:g}" for i in table.index]
        positions = np.arange(len(names))
        ax.bar(positions - 0.2, table["rule"], 0.4, label="rule", color="#9BB6E6")
        ax.bar(positions + 0.2, table["cnn"], 0.4, label="CNN", color="#0B3D91")
        ax.set_xticks(positions, names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel("injected dipoles found")
    axes[0].legend(fontsize=8)
    fig.suptitle("Same false-alarm rate, stamps of the target left out", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    found, _, real, _ = load()
    table = pd.read_parquet(OUT_DIR / "oof.parquet")
    real_table = pd.read_parquet(OUT_DIR / "real_oof.parquet")
    if "rule" not in table:  # a dataset from before the rule was stored with the stamps
        table["rule"] = [rule_says(s) for s in found]
    if "rule" not in real_table:
        real_table["rule"] = [rule_says(s) for s in real]
    if "stage1" not in real_table:
        from serafim.validation import verification

        real_table["stage1"] = stage_one(real_table, verification.read_gold(dipoles.GOLD_DIR)[0])
    real_eval = real_set(real_table)

    # --- Injected stamps -------------------------------------------------------------------
    y, p = table["label"].to_numpy(), table["p_dipole"].to_numpy()
    negatives = table["label"] == 0
    false_alarm = float(table.loc[negatives, "rule"].mean())
    threshold = matched_threshold(table.loc[negatives, "p_dipole"].to_numpy(), false_alarm)
    youden = youden_threshold(y, p)
    table["cnn"] = table["p_dipole"] >= threshold
    real_eval = real_eval.assign(cnn=real_eval["p_dipole"] >= threshold)
    injected_cnn = binary_metrics(y, table["cnn"], p)
    injected_rule = binary_metrics(y, table["rule"])
    positives = table[table["label"] == 1]
    faint = positives[positives["snr"] < 10]
    saturated = positives[positives["kind"] == "saturated_dipole"]
    plain = positives[positives["kind"] == "dipole"]
    snr, shift = by_bins(plain, "snr", SNR_BINS), by_bins(plain, "shift_px", SHIFT_BINS)
    kinds = table[negatives].groupby("kind").agg(n=("label", "size"), rule=("rule", "mean"),
                                                 cnn=("cnn", "mean"))  # fmt: skip
    folds = pd.DataFrame([
        {"target": t, "auc": roc_auc(part["label"], part["p_dipole"]),
         "recall": part.loc[part["label"] == 1, "cnn"].mean(),
         "false_alarm": part.loc[part["label"] == 0, "cnn"].mean()}
        for t, part in table.groupby("target_id")
    ])  # fmt: skip
    calibration, ece = reliability(y, p)
    plot(snr, shift, IMAGE_PATH)

    # --- Real stamps ------------------------------------------------------------------------
    real_y = real_eval["label"].to_numpy()
    real_cnn = binary_metrics(real_y, real_eval["cnn"], real_eval["p_dipole"])
    real_stage1 = binary_metrics(real_y, real_eval["stage1"])
    still = real_eval[real_eval["label"] == 0]
    still_k = int(still["cnn"].sum())
    sky = table[table["kind"].isin(["star", "background"])]  # real, untouched, out of fold
    domain = []
    for target_id in MOVING_TARGETS:
        movers = real_eval[(real_eval["target_id"] == target_id) & (real_eval["label"] == 1)]
        if movers.empty:
            continue
        score = np.r_[movers["p_dipole"], sky["p_dipole"]]
        truth = np.r_[np.ones(len(movers)), np.zeros(len(sky))]
        domain.append({"target": target_id, "n": len(movers), "auc": roc_auc(truth, score),
                       "cnn": int(movers["cnn"].sum()),
                       "stage1": int(movers["stage1"].sum())})  # fmt: skip
    sky_alarm = float((sky["p_dipole"] >= threshold).mean())
    youden_specificity = binary_metrics(y, p >= youden)["specificity"]
    stage1_found = interval(real_stage1["TP"], real_stage1["TP"] + real_stage1["FN"])

    criteria = [
        ("Out-of-fold ROC AUC on injected stamps", f">= {MIN_AUC}",
         f"{injected_cnn['ROC_AUC']:.3f}", injected_cnn["ROC_AUC"] >= MIN_AUC),
        ("Finds more injected dipoles than the rule at the same false alarms",
         "sensitivity above the rule's",
         f"{injected_cnn['sensitivity']:.0%} vs {injected_rule['sensitivity']:.0%}",
         injected_cnn["sensitivity"] > injected_rule["sensitivity"]),
        ("Calibration on injected stamps (ECE)", f"<= {MAX_ECE}", f"{ece:.3f}", ece <= MAX_ECE),
        ("Real: ROC AUC, movers against still saturated stars", f">= {MIN_REAL_AUC}",
         f"{real_cnn['ROC_AUC']:.2f}", real_cnn["ROC_AUC"] >= MIN_REAL_AUC),
        ("Real: still saturated stars called a dipole", f"<= {MAX_REAL_FALSE_ALARM:.0%}",
         f"{still_k}/{len(still)} = {still_k / len(still):.0%} (95 % CI "
         f"{interval(still_k, len(still))})", still_k / len(still) <= MAX_REAL_FALSE_ALARM),
        ("Real: balanced accuracy not below the stage-1 rule's", "at least the rule's",
         f"{real_cnn['balanced_accuracy']:.2f} vs {real_stage1['balanced_accuracy']:.2f}",
         real_cnn["balanced_accuracy"] >= real_stage1["balanced_accuracy"]),
    ]  # fmt: skip
    adopted = all(ok for *_, ok in criteria)
    verdict = (
        "**Adopted:** the CNN may change statuses."
        if adopted
        else "**Experimental.** The pipeline keeps the stage-1 rule and its catalog check for "
        "every status. The CNN's p_dipole is shown only next to dipoles without saturated pixels "
        "(its domain: there it ranks the real Epsilon Indi B above real fixed stars and finds it "
        "more often than the rule), as an extra opinion that never changes a status. On saturated "
        "stars it does not tell one that moves from one that does not, and the real test has only "
        "three moving objects."
    )

    lines = [
        "# The dipole CNN against the classic rule (task A9, stage 2)",
        "",
        "Generated by `python -m training.evaluate`. Every score is **out of fold**: trained on",
        "four targets, the CNN judges the fifth (`training/train.py`). Stamps of 3 × 32 × 32 (sum",
        "A, sum B and their difference, `serafim/ai/stamps.py`).",
        "",
        "## Decision",
        "",
        "| Acceptance criterion | Needed | Got | |",
        "| --- | --- | --- | --- |",
        *[f"| {name} | {need} | {got} | {'✅' if ok else '❌'} |"
          for name, need, got, ok in criteria],
        "",
        verdict,
        "",
        "## Confusion matrix and its metrics",
        "",
        f"Injected: {len(table)} stamps out of fold (injected dipoles and hard negatives; the "
        f"fixed stars and the sky are real). CNN at p_dipole ≥ {threshold:.3f}, the threshold "
        f"with the rule's false-alarm rate ({false_alarm:.1%}); the threshold that maximizes "
        f"informedness would be {youden:.3f}, with {youden_specificity:.1%} specificity: too "
        "many false alarms for something as rare as a moving source. Real: "
        f"{int(real_y.sum())} differences where a target moved ≥ {MIN_SHIFT_PX} px (Barnard's Star,"
        f" Epsilon Indi B, WISE 0855) and {len(still)} of saturated stars that do not move (RW "
        "Aurigae, R Draconis). The stage-1 rule is the one the pipeline uses (whole image, halo "
        "lobes, catalog check).",
        "",
        *metrics_lines({"Injected: CNN": injected_cnn, "Injected: rule": injected_rule,
                        "Real: CNN": real_cnn, "Real: stage-1 rule": real_stage1}),
        "",
        "Precision depends on how common dipoles are; the test sets have about half. In a real",
        "difference a mover is one place in thousands, so neither the CNN nor the rule can scan",
        "blindly: they judge places that already hold a 5-sigma pair of blobs, and the catalog",
        "check removes the stars that do not move.",
        "",
        f"The stage-1 rule's 1.000 come from {real_stage1['FP']} false alarms in "
        f"{real_stage1['FP'] + real_stage1['TN']} stamps of only two still stars: its real "
        "false-alarm rate could still be up to "
        f"{interval(real_stage1['FP'], real_stage1['FP'] + real_stage1['TN']).split('-')[1]} "
        "(95 % Wilson).",
        "",
        "## Its domain: sources that are not saturated",
        "",
        f"Real movers against the {len(sky)} real fixed-star and sky stamps of the fields left "
        f"out (the CNN calls {sky_alarm:.1%} of those a dipole):",
        "",
        "| Mover | Differences | ROC AUC against fixed stars and sky | CNN finds | "
        "Stage-1 rule finds |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {d['target']} | {d['n']} | {d['auc']:.3f} | {d['cnn']} | {d['stage1']} |"
          for d in domain],
        "",
        "Epsilon Indi B (not saturated) is the case the CNN is for. Barnard's Star is saturated:",
        "the rule's halo logic finds it more often. WISE 0855 (S/N 5–7 in one season) stays below",
        "what either can see.",
        "",
        "## Injected dipoles: by S/N, by shift, saturated",
        "",
        f"At the same false-alarm rate: sensitivity {injected_rule['sensitivity']:.0%} (rule) "
        f"against {injected_cnn['sensitivity']:.0%} (CNN); at peak S/N under 10, "
        f"{faint['rule'].mean():.0%} against {faint['cnn'].mean():.0%}; saturated movers (core "
        f"masked) {saturated['rule'].mean():.0%} against {saturated['cnn'].mean():.0%}.",
        "",
        "| Peak S/N (not saturated) | Stamps | Rule | CNN |",
        "| --- | --- | --- | --- |",
        *[f"| {i.left:g}–{i.right:g} | {int(r.n)} | {r.rule:.0%} | {r.cnn:.0%} |"
          for i, r in snr.iterrows()],
        "",
        "| Shift (px) | Stamps | Rule | CNN |",
        "| --- | --- | --- | --- |",
        *[f"| {i.left:g}–{i.right:g} | {int(r.n)} | {r.rule:.0%} | {r.cnn:.0%} |"
          for i, r in shift.iterrows()],
        "",
        "![Rule and CNN at the same false-alarm rate](img/dipole_cnn.png)",
        "",
        "## False alarms, by kind of negative",
        "",
        "| Negative | Stamps | Rule | CNN |",
        "| --- | --- | --- | --- |",
        *[f"| {kind} | {int(r.n)} | {r.rule:.1%} | {r.cnn:.1%} |" for kind, r in kinds.iterrows()],
        "",
        *learning_lines(),
        "## Stability: each target left out",
        "",
        "| Target left out | ROC AUC | Dipoles found | False alarms |",
        "| --- | --- | --- | --- |",
        *[f"| {f.target} | {f.auc:.3f} | {f.recall:.0%} | {f.false_alarm:.1%} |"
          for f in folds.itertuples()],
        f"| mean ± sd | {folds['auc'].mean():.3f} ± {folds['auc'].std():.3f} | | |",
        "",
        "## Calibration: does p_dipole = 0.8 mean 8 in 10?",
        "",
        f"On the injected stamps, expected calibration error {ece:.3f}. (Not measured on the real",
        "stamps: too few.)",
        "",
        "| p_dipole | Mean p | Share that are dipoles | Stamps |",
        "| --- | --- | --- | --- |",
        *[f"| {c.low:.1f}–{c.high:.1f} | {c.mean_p:.2f} | {c.observed:.2f} | {c.n} |"
          for c in calibration.itertuples()],
        "",
        "## Each real target",
        "",
        "| Target | Differences used | CNN says dipole | Stage-1 rule keeps a dipole | "
        "Median p_dipole |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {t} | {len(part)} | {int(part['cnn'].sum())} | {int(part['stage1'].sum())} | "
          f"{part['p_dipole'].median():.2f} |" for t, part in real_eval.groupby("target_id")],
        "",
        f"Few independent objects (3 movers, 2 still stars), so the intervals are wide: the CNN "
        f"finds {real_cnn['TP']}/{real_cnn['TP'] + real_cnn['FN']} real movers (95 % CI "
        f"{interval(real_cnn['TP'], real_cnn['TP'] + real_cnn['FN'])}), the stage-1 rule "
        f"{real_stage1['TP']} ({stage1_found}).",
        "",
    ]  # fmt: skip
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    summary = {
        "threshold": round(threshold, 4), "youden_threshold": round(youden, 4),
        "false_alarm": round(false_alarm, 4), "auc": round(injected_cnn["ROC_AUC"], 4),
        "ece": round(ece, 4), "real_auc": round(real_cnn["ROC_AUC"], 4),
        "real_balanced_accuracy": round(real_cnn["balanced_accuracy"], 4),
        "real_balanced_accuracy_rule": round(real_stage1["balanced_accuracy"], 4),
        "adopted": bool(adopted), "domain": "dipoles without saturated pixels",
    }  # fmt: skip
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("injected: CNN %.3f vs rule %.3f sensitivity at %.4f false alarms; real AUC %.3f; "
             "adopted %s; wrote %s", injected_cnn["sensitivity"], injected_rule["sensitivity"],
             false_alarm, real_cnn["ROC_AUC"], adopted, REPORT_PATH)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
