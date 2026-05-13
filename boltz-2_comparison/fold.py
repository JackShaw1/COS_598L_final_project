import csv
import io
import json
import shutil
import statistics
import tarfile
import urllib.request
from collections import deque
from pathlib import Path
import modal

"""
This script produces Boltz-2 folds for 72 random positive samples and 20 controls
Run using "modal boltz-2_comparison/fold.py"
Please note that this script has a warming cycle that is important for running large jobs via Modal 
"""

# Make so others can repurpose (this script can be used elsewhere as well)
LIGAND_NAME = "Imatinib"
LIGAND_SMILES = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"

CANDIDATES_CSV = Path(__file__).parent / "prots.csv"

# Randomly selected uniprot Ids (chosen using Claude)
CONTROL_UNIPROT_IDS = [
    "P02768", "P02787", "P02649", "P02766", "P04637",
    "P00734", "P00915", "P00918", "P29401", "P04406",
    "P00558", "P14618", "P68133", "P10809", "P11142",
    "P00367", "P00387", "P22392", "P10599", "P02545",
]

# Requesting 10 GPUs in parallel
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

OUTPUT_DIR = Path("boltz-2_comparison/boltz2_imatinib_outputs")
SUMMARY_JSON = Path("boltz-2_comparison/imatinib_results.json")

CACHE_DIR = Path("/models/boltz")
CACHE_WARMUP_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
CACHE_WARMUP_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

def load_candidate_ids():
    ids = []
    seen = set()

    with CANDIDATES_CSV.open() as fh:
        for row in csv.DictReader(fh):
            uid = row["uniprot_id"].strip()
            if uid and uid not in seen:
                seen.add(uid)
                ids.append(uid)

    return ids

# Gather sequence from uniprot
def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    with urllib.request.urlopen(url, timeout=30) as fh:
        fasta = fh.read().decode()

    return "".join(fasta.splitlines()[1:]).strip()

# Create Boltz-2 upload
def make_yaml(sequence, smiles, empty_msa=False):
    msa_line = "\n      msa: empty" if empty_msa else ""

    return f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}{msa_line}
  - ligand:
      id: B
      smiles: {json.dumps(smiles)}
properties:
  - affinity:
      binder: B
"""

# Run Boltz-2 command
def boltz_command(input_path, out_dir, warmup=False):
    if warmup:
        recycling_steps = 1
        sampling_steps = 50
        diffusion_samples = 1
        sampling_steps_affinity = 50
        diffusion_samples_affinity = 1
    else:
        recycling_steps = RECYCLING_STEPS
        sampling_steps = SAMPLING_STEPS
        diffusion_samples = DIFFUSION_SAMPLES
        sampling_steps_affinity = SAMPLING_STEPS_AFFINITY
        diffusion_samples_affinity = DIFFUSION_SAMPLES_AFFINITY

    cmd = [
        "boltz", "predict", str(input_path),
        "--cache", str(CACHE_DIR),
        "--out_dir", str(out_dir),
        "--accelerator", "gpu",
        "--recycling_steps", str(recycling_steps),
        "--sampling_steps", str(sampling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--max_parallel_samples", str(MAX_PARALLEL_SAMPLES),
        "--num_workers", str(NUM_WORKERS),
        "--sampling_steps_affinity", str(sampling_steps_affinity),
        "--diffusion_samples_affinity", str(diffusion_samples_affinity),
        "--model", "boltz2",
        "--override",
    ]

    if USE_MSA_SERVER and not warmup:
        cmd.append("--use_msa_server")
        cmd.extend(["--msa_server_url", MSA_SERVER_URL])

    return cmd

def run_command(cmd, label, tail_lines=160):
    import subprocess

    tail = deque(maxlen=tail_lines)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())

    return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"Last {len(tail)} output lines:\n"
            f"{chr(10).join(tail)}"
        )

def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

def tar_folder(path):
    bio = io.BytesIO()

    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for file in sorted(path.rglob("*")):
            tf.add(file, arcname=file.relative_to(path))

    return bio.getvalue()

def extract_tar(blob, destination):
    reset_dir(destination)

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(destination)

def read_first_json(folder, pattern):
    matches = sorted(folder.rglob(pattern))

    if not matches:
        return {}

    return json.loads(matches[0].read_text())

def blank_summary_row(job_name, meta):
    return {
        "job_name": job_name,
        "uniprot_id": meta.get("uniprot_id"),
        "group": meta.get("group"),
        "sequence_length": meta.get("sequence_length"),
        "affinity_pred_value": None,
        "affinity_probability_binary": None,
        "iptm": None,
        "ptm": None,
        "complex_plddt": None,
    }

def fmt(row, key):
    value = row.get(key)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"

def score(row):
    value = row.get("affinity_probability_binary")
    return value if isinstance(value, (int, float)) else -1


# MODAL APP
app = modal.App(name="boltz2-imatinib-screen")

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

boltz_cache_volume = modal.Volume.from_name("boltz-models", create_if_missing=True)


@app.function(
    image=image,
    volumes={CACHE_DIR: boltz_cache_volume},
    gpu=GPU,
    max_containers=1,
    timeout=2 * 60 * 60,
)
def warm_boltz_cache():
    work_dir = Path("/tmp/boltz_cache_warmup")
    input_path = work_dir / "cache_warmup.yaml"
    out_dir = work_dir / "out"

    reset_dir(work_dir)

    input_path.write_text(
        make_yaml(
            CACHE_WARMUP_SEQUENCE,
            CACHE_WARMUP_SMILES,
            empty_msa=True,
        )
    )

    print("Warming Boltz cache in a single container.")
    run_command(boltz_command(input_path, out_dir, warmup=True), "Boltz cache warmup")
    boltz_cache_volume.commit()
    print("Boltz cache warmup finished and committed.")


@app.function(
    image=image,
    volumes={CACHE_DIR: boltz_cache_volume},
    gpu=GPU,
    max_containers=MAX_CONTAINERS,
    timeout=24 * 60 * 60,
)
def run_one_boltz_job(job):
    job_name = job["job_name"]

    boltz_cache_volume.reload()

    work_dir = Path("/tmp") / f"boltz_{job_name}"
    input_path = work_dir / f"{job_name}.yaml"
    out_dir = work_dir / "out"

    reset_dir(work_dir)
    input_path.write_text(job["yaml_text"])

    print(f"Starting {job_name}")

    try:
        run_command(boltz_command(input_path, out_dir), job_name)
    except Exception as exc:
        return {
            "job_name": job_name,
            "ok": False,
            "error": str(exc),
        }

    print(f"Finished {job_name}")

    return {
        "job_name": job_name,
        "ok": True,
        "tar_gz": tar_folder(out_dir),
    }

# Local
@app.local_entrypoint()
def main():
    output_dir = OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids = load_candidate_ids()
    candidate_set = set(candidate_ids)

    control_ids = [
        uid for uid in CONTROL_UNIPROT_IDS
        if uid not in candidate_set
    ]

    overlap = candidate_set & set(CONTROL_UNIPROT_IDS)

    if overlap:
        print(f"Dropping controls that overlap candidates: {sorted(overlap)}")

    print(
        f"Loaded {len(candidate_ids)} unique candidate UniProt IDs and "
        f"{len(control_ids)} control UniProt IDs."
    )
    print(f"Ligand: {LIGAND_NAME}  SMILES: {LIGAND_SMILES}")

    proteins = (
        [("candidate", f"cand_{uid}", uid) for uid in candidate_ids]
        + [("control", f"ctrl_{uid}", uid) for uid in control_ids]
    )

    jobs = []
    job_meta = {}

    print("\nFetching protein sequences from UniProt...")

    for group, job_name, uid in proteins:
        job_meta[job_name] = {
            "group": group,
            "uniprot_id": uid,
            "sequence_length": None,
        }

        try:
            sequence = fetch_sequence(uid)
        except Exception as exc:
            print(f"  {job_name} ({uid}): FAILED to fetch sequence ({exc})")
            job_meta[job_name]["fetch_error"] = str(exc)
            continue

        job_meta[job_name]["sequence_length"] = len(sequence)
        jobs.append({
            "job_name": job_name,
            "yaml_text": make_yaml(sequence, LIGAND_SMILES),
        })

        print(f"  {job_name} ({uid}, {group}): {len(sequence)} aa")

    print("\nWarming Boltz model cache before launching parallel jobs...")
    warm_boltz_cache.remote()
    print("Cache warmup complete.")

    print(f"\nSubmitting {len(jobs)} parallel Boltz-2 jobs on {GPU}...")

    summary = []

    for result in run_one_boltz_job.map(jobs, order_outputs=False):
        job_name = result["job_name"]
        row = blank_summary_row(job_name, job_meta[job_name])

        if not result.get("ok"):
            row["error"] = result.get("error", "Unknown Boltz failure")
            summary.append(row)
            print(f"  {job_name}: FAILED\n{row['error']}")
            continue

        job_out = output_dir / job_name
        extract_tar(result["tar_gz"], job_out)

        affinity = read_first_json(job_out, f"affinity_{job_name}.json")
        confidence = read_first_json(job_out, f"confidence_{job_name}_model_*.json")

        row["affinity_pred_value"] = affinity.get("affinity_pred_value")
        row["affinity_probability_binary"] = affinity.get("affinity_probability_binary")
        row["iptm"] = confidence.get("iptm")
        row["ptm"] = confidence.get("ptm")
        row["complex_plddt"] = confidence.get("complex_plddt")

        summary.append(row)

        print(
            f"{job_name}:"
            f"P(binder)={row['affinity_probability_binary']}"
        )

    completed = {row["job_name"] for row in summary}

    for job_name, meta in job_meta.items():
        if job_name in completed:
            continue

        row = blank_summary_row(job_name, meta)
        row["error"] = meta.get("fetch_error", "Sequence fetch failed")
        summary.append(row)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    failures = [row for row in summary if row.get("error")]

    if failures:
        print(f"\n{len(failures)} job(s) failed or were skipped:")