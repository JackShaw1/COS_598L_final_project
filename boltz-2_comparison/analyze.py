import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path("boltz-2_comparison")
CSV_PATH = ROOT / "prots.csv"
OUTPUTS_DIR = ROOT / "boltz2_imatinib_outputs"
SCATTER_PNG = ROOT / "classifier_vs_boltz_scatter.png"
HIST_PNG = ROOT / "pbinder_candidates_vs_controls_hist.png"

"""
This scripts creates the two plots for Figure 3:
  1. Scatter: classifier score vs. Boltz P(binder), candidates only.
  2. Overlaid histogram: P(binder) for candidates vs. controls.

And it reads from:
- boltz-2_comparison/prots.csv
- boltz-2_comparison/boltz2_imatinib_outputs/<job_name>/.../affinity_<job_name>.json

Output files:
- boltz-2_comparison/classifier_vs_boltz_scatter.png
- boltz-2_comparison/pbinder_candidates_vs_controls_hist.png
"""

# Give per protein maxixmum classifier scores
def load_classifier_scores(csv_path):
    per_protein: dict[str, float] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row["uniprot_id"].strip()
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if uid not in per_protein or score > per_protein[uid]:
                per_protein[uid] = score
    return per_protein

def load_boltz_results(outputs_dir):
    """
    Walk the Boltz output tree and return {uniprot_id: {group, p_binder, log10_ic50_uM}}.
    Job folder names are `cand_<UNIPROT>` or `ctrl_<UNIPROT>`.
    """
    results: dict[str, dict] = {}
    if not outputs_dir.exists():
        return results
    for job_dir in sorted(outputs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        name = job_dir.name
        if name.startswith("cand_"):
            group, uid = "candidate", name[len("cand_"):]
        elif name.startswith("ctrl_"):
            group, uid = "control", name[len("ctrl_"):]
        else:
            continue

        aff_files = list(job_dir.rglob(f"affinity_{name}.json"))
        if not aff_files:
            continue
        try:
            data = json.loads(aff_files[0].read_text())
        except (OSError, json.JSONDecodeError):
            continue

        results[uid] = {
            "group": group,
            "p_binder": data.get("affinity_probability_binary"),
            "log10_ic50_uM": data.get("affinity_pred_value"),
        }
    return results

def main():
    classifier = load_classifier_scores(CSV_PATH)
    boltz = load_boltz_results(OUTPUTS_DIR)

    print(f"Classifier scores: {len(classifier)} unique candidate UniProt IDs")
    print(f"Boltz folds found: {len(boltz)} (across both groups)")

    cand_uids = [
        uid for uid, r in boltz.items()
        if r["group"] == "candidate"
        and isinstance(r["p_binder"], (int, float))
        and uid in classifier
    ]
    cand_classifier = np.array([classifier[uid] for uid in cand_uids])
    cand_pbinder = np.array([boltz[uid]["p_binder"] for uid in cand_uids])

    cand_pbinder_all = np.array([
        r["p_binder"] for r in boltz.values()
        if r["group"] == "candidate" and isinstance(r["p_binder"], (int, float))
    ])
    ctrl_pbinder = np.array([
        r["p_binder"] for r in boltz.values()
        if r["group"] == "control" and isinstance(r["p_binder"], (int, float))
    ])

    print(
        f"\nPaired (classifier, Boltz) candidates: {len(cand_uids)}"
        f"  | Boltz candidates total: {len(cand_pbinder_all)}"
        f"  | Boltz controls: {len(ctrl_pbinder)}"
    )

    # Scatter for classifier score vs Boltz P(binder)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(cand_classifier, cand_pbinder, alpha=0.75, edgecolor="black", linewidth=0.4)
    for x, y, uid in zip(cand_classifier, cand_pbinder, cand_uids):
        ax.annotate(uid, (x, y), fontsize=6, alpha=0.55, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Boltz-2 affinity-based interaction probabilities")
    ax.set_ylim(0,.75)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=160)
    plt.close(fig)
    print(f"\nWrote scatter: {SCATTER_PNG}")

    # Histogram for candidates vs controls P(binder) overlaid
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bins = np.linspace(0.0, 1.0, 21)
    if len(cand_pbinder_all):
        ax.hist(
            cand_pbinder_all, bins=bins, alpha=0.55,
            label=f"Candidates (n={len(cand_pbinder_all)}, mean={cand_pbinder_all.mean():.2f})",
            color="C0", edgecolor="black", linewidth=0.4, density=True,
        )
    if len(ctrl_pbinder):
        ax.hist(
            ctrl_pbinder, bins=bins, alpha=0.55,
            label=f"Controls (n={len(ctrl_pbinder)}, mean={ctrl_pbinder.mean():.2f})",
            color="C3", edgecolor="black", linewidth=0.4, density=True,
        )
    ax.set_xlabel("Boltz-2 affinity-based interaction probabilities")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HIST_PNG, dpi=160)
    plt.close(fig)
    print(f"Wrote histogram: {HIST_PNG}")

if __name__ == "__main__":
    main()
