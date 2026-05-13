import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
# Modal serverless GPUs
import modal

"""
This scripts folds additional control samples using the same approach as boltz-2_comparison/fold.py. 
These control proteins were randomly sampled from the entire human proteome using Claude.
"""

LIGAND_NAME = "Imatinib"
LIGAND_SMILES = ("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")

# 20 additional diverse, well-characterized human Swiss-Prot proteins chosen using Claude
EXTRA_CONTROL_UNIPROT_IDS = [
    "P02675",  # Fibrinogen beta chain (FGB)
    "P02679",  # Fibrinogen gamma chain (FGG)
    "P00738",  # Haptoglobin (HP)
    "P02763",  # Alpha-1-acid glycoprotein 1 (ORM1)
    "P02753",  # Retinol-binding protein 4 (RBP4)
    "P02749",  # Beta-2-glycoprotein 1 (APOH)
    "P02652",  # Apolipoprotein A-II (APOA2)
    "P02647",  # Apolipoprotein A-I (APOA1)
    "P00390",  # Glutathione reductase (GSR)
    "P00505",  # Aspartate aminotransferase, mitochondrial (GOT2)
    "P09382",  # Galectin-1 (LGALS1)
    "P06748",  # Nucleophosmin (NPM1)
    "P00441",  # Superoxide dismutase 1 (SOD1)
    "P07900",  # HSP90AA1
    "P25705",  # ATP synthase F1 alpha (ATP5F1A)
    "P06576",  # ATP synthase F1 beta (ATP5F1B)
    "P04075",  # Fructose-bisphosphate aldolase A (ALDOA)
    "P30086",  # Phosphatidylethanolamine-binding protein 1 (PEBP1)
    "P63104",  # 14-3-3 zeta (YWHAZ)
    "P08758",  # Annexin A5 (ANXA5)
]

LOCAL_OUTPUTS_DIR = Path("boltz-2_comparison/boltz2_imatinib_outputs")

# Get 10 A100s in parallel
GPU = "A100-80GB"
MAX_CONTAINERS = 10

USE_MSA_SERVER = True
MSA_SERVER_URL = "https://api.colabfold.com"

RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
MAX_PARALLEL_SAMPLES = 1
NUM_WORKERS = 1
SAMPLING_STEPS_AFFINITY = 200
DIFFUSION_SAMPLES_AFFINITY = 5

SUMMARY_JSON = Path("boltz-2_comparison") / "imatinib_extra_controls_results.json"

# Gather sequence from uniprot
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    with urllib.request.urlopen(url, timeout=30) as fh:
        text = fh.read().decode()
    return "".join(text.splitlines()[1:]).strip()

# Create Boltz-2 upload
def build_yaml(sequence, smiles):
    return f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
  - ligand:
      id: B
      smiles: {json.dumps(smiles)}
properties:
  - affinity:
      binder: B
"""

def extract_targz_bytes(blob, destination):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(destination)

# Run boltz-2 command
def run_command_streaming(cmd: list[str], label: str, tail_lines: int = 160) -> None:
    import subprocess
    from collections import deque

    tail = deque(maxlen=tail_lines)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())

    return_code = proc.wait()
    if return_code != 0:
        cmd_text = " ".join(cmd)
        tail_text = "\n".join(tail)
        raise RuntimeError(
            f"{label} failed with exit code {return_code}.\n"
            f"Command: {cmd_text}\n\n"
            f"Last {len(tail)} output lines:\n{tail_text}"
        )

# Modal interaction
app = modal.App(name="boltz2-imatinib-extra-controls")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "boltz==2.1.1",
        "torch==2.6.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
        extra_options="--index-strategy unsafe-best-match",
    )
)

# Reuse the same model volume as fold.py — Boltz weights are already cached
boltz_cache_volume = modal.Volume.from_name("boltz-models", create_if_missing=True)
CACHE_DIR = Path("/models/boltz")

@app.function(
    image=image,
    volumes={CACHE_DIR: boltz_cache_volume},
    gpu=GPU,
    max_containers=MAX_CONTAINERS,
    timeout=24 * 60 * 60,
)
def run_one_boltz_job(job: dict) -> dict:
    import io
    import shutil
    import tarfile
    from pathlib import Path

    job_name = job["job_name"]
    yaml_text = job["yaml_text"]

    boltz_cache_volume.reload()

    work_dir = Path("/tmp") / f"boltz_{job_name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = work_dir / f"{job_name}.yaml"
    out_dir = work_dir / "out"
    input_path.write_text(yaml_text)

    cmd = [
        "boltz", "predict", str(input_path),
        "--cache", str(CACHE_DIR),
        "--out_dir", str(out_dir),
        "--accelerator", "gpu",
        "--recycling_steps", str(RECYCLING_STEPS),
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--max_parallel_samples", str(MAX_PARALLEL_SAMPLES),
        "--num_workers", str(NUM_WORKERS),
        "--sampling_steps_affinity", str(SAMPLING_STEPS_AFFINITY),
        "--diffusion_samples_affinity", str(DIFFUSION_SAMPLES_AFFINITY),
        "--model", "boltz2",
        "--override",
    ]
    if USE_MSA_SERVER:
        cmd.append("--use_msa_server")
        if MSA_SERVER_URL:
            cmd.extend(["--msa_server_url", MSA_SERVER_URL])

    print(f"Starting {job_name}")
    try:
        run_command_streaming(cmd, job_name)
    except Exception as exc:
        return {"job_name": job_name, "ok": False, "error": str(exc)}
    print(f"Finished {job_name}")

    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for path in sorted(out_dir.rglob("*")):
            tf.add(path, arcname=path.relative_to(out_dir))

    return {"job_name": job_name, "ok": True, "tar_gz": bio.getvalue()}

# Local entrypoint for modal app
@app.local_entrypoint()
def main():
    output_dir = LOCAL_OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extra controls to fold: {len(EXTRA_CONTROL_UNIPROT_IDS)}")

    # add to folds without existing
    to_fold: list[str] = []
    for uid in EXTRA_CONTROL_UNIPROT_IDS:
        existing = output_dir / f"ctrl_{uid}"
        aff_files = list(existing.rglob(f"affinity_ctrl_{uid}.json")) if existing.exists() else []
        if aff_files:
            continue
        else:
            to_fold.append(uid)

    jobs = []
    job_meta: dict[str, dict] = {}
    for uid in to_fold:
        job_name = f"ctrl_{uid}"
        try:
            seq = fetch_uniprot_sequence(uid)
        except Exception as exc:
            job_meta[job_name] = {"uniprot_id": uid, "fetch_error": str(exc)}
            continue
        print(f"  {job_name} ({uid}): {len(seq)} aa")
        job_meta[job_name] = {"uniprot_id": uid, "sequence_length": len(seq)}
        jobs.append({"job_name": job_name, "yaml_text": build_yaml(seq, LIGAND_SMILES)})

    print(f"\nSubmitting {len(jobs)} parallel Boltz-2 jobs on {GPU}...")
    summary: list[dict] = []
    for result in run_one_boltz_job.map(jobs, order_outputs=False):
        name = result["job_name"]
        meta = job_meta.get(name, {})
        row = {
            "job_name": name,
            "uniprot_id": meta.get("uniprot_id"),
            "group": "control",
            "sequence_length": meta.get("sequence_length"),
            "affinity_pred_value": None,
            "affinity_probability_binary": None,
            "iptm": None,
            "ptm": None,
            "complex_plddt": None,
        }

        if not result.get("ok", True):
            row["error"] = result.get("error", "Unknown Boltz failure")
            summary.append(row)
            continue

        job_out = output_dir / name
        extract_targz_bytes(result["tar_gz"], job_out)

        aff_path = next(iter(sorted(job_out.rglob(f"affinity_{name}.json"))), None)
        conf_path = next(iter(sorted(job_out.rglob(f"confidence_{name}_model_*.json"))), None)
        aff = json.loads(aff_path.read_text()) if aff_path else {}
        conf = json.loads(conf_path.read_text()) if conf_path else {}

        row["affinity_pred_value"] = aff.get("affinity_pred_value")
        row["affinity_probability_binary"] = aff.get("affinity_probability_binary")
        row["iptm"] = conf.get("iptm")
        row["ptm"] = conf.get("ptm")
        row["complex_plddt"] = conf.get("complex_plddt")
        summary.append(row)
        print(
            f"{name}: "
            f"P(binder)={row['affinity_probability_binary']}"
        )

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
