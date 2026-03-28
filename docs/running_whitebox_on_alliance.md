# Running the White-Box Pipeline on Alliance Canada

This guide explains how to run the white-box cross-model probe on an Alliance Canada cluster (e.g. **Killarney**, Vulcan, or Nibi).

---

## 1. Prerequisites

- **CCDB account** and access to an AI cluster (e.g. Killarney; may require AIP-type RAP from your AI institution).
- **SSH** to the login node, e.g.:
  ```bash
  ssh USERNAME@killarney.alliancecan.ca
  ```
- **Data**: the evaluated JSONL and (optionally) this repo on the cluster.

---

## 2. Transfer Your Project and Data

From your **local machine**:

```bash
# Option A: Clone from Git (if repo is on GitHub/GitLab)
cd $SCRATCH   # or your project dir on the cluster
git clone https://github.com/YOUR_USER/LLM_Hallucination_Measure.git
cd LLM_Hallucination_Measure
```

If the repo is only on your laptop, use **rsync** or **scp** to copy it to the cluster:

```bash
# From your laptop (replace USER and CLUSTER)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/simar/LLM_Hallucination_Measure/ \
  USER@killarney.alliancecan.ca:$SCRATCH/LLM_Hallucination_Measure/
```

Ensure the **evaluated JSONL** is on the cluster. The script default is:

```
.../data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.jsonl
```

If your file has a different name (e.g. `...analysis_ready.skip_greedy_semantic_eval.jsonl`), set `INPUT_FILE` when submitting (see below).

---

## 3. Set Up the Environment on the Cluster

SSH to the cluster, then:

```bash
cd $SCRATCH/LLM_Hallucination_Measure   # or your path

# Load Python (adjust for your cluster; examples below)
module load python/3.10    # if available
# or
module load miniconda3     # then conda activate base

# Create a virtual environment (recommended)
python3 -m venv venv_wb
source venv_wb/bin/activate

# Install dependencies (PyTorch with CUDA – check cluster docs for recommended version)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # example for CUDA 12.1
pip install -r requirements.txt
pip install transformers accelerate
```

**HuggingFace token** (needed for gated models like Llama):

```bash
# Option 1: environment variable (add to your ~/.bashrc or set in the job)
export HF_TOKEN="hf_xxxxxxxx"

# Option 2: .env in project root (do not commit this file)
echo "HF_TOKEN=hf_xxxxxxxx" >> .env
```

---

## 4. Request the Right Resources

On **Killarney**:

- **Standard tier**: 4× L40S (48 GB each) per node. Use `--gres=gpu:4` and `--sequential-encoders` so only one big model is in memory at a time.
- **Performance tier**: 8× H100 (80 GB each). You can use 1–2 GPUs for 80B models; adjust `#SBATCH --gres=gpu:2` (or 4) in the script.

Edit the Slurm script (see below) and set:

- `#SBATCH --account=def-<your_ccdb_account>`
- Optionally `#SBATCH --mail-user=your@email.ca`

---

## 5. Run One Job (Single Target + Verifier)

From the **repo root** on the cluster:

```bash
cd $SCRATCH/LLM_Hallucination_Measure
source venv_wb/bin/activate   # if using venv

# Default in script: DeepSeek V3.2 (DeepSeek) as target/response, Qwen3 Next 80B as verifier
# TARGET_NAME must match the "model" field in your JSONL exactly.
sbatch scripts/slurm_run_wb_probe.sh
```

To run a **different (target, response, verifier)** without editing the script:

```bash
sbatch --export=TARGET_NAME="Qwen3 Next 80B (OpenRouter)",RESPONSE_HF="Qwen/Qwen3-Next-80B",VERIFIER_HF="deepseek-ai/DeepSeek-V3" \
  scripts/slurm_run_wb_probe.sh
```

If your input file is elsewhere:

```bash
sbatch --export=INPUT_FILE="$SCRATCH/my_data/results.jsonl" scripts/slurm_run_wb_probe.sh
```

---

## 6. Run All Three Response-Model Combinations

Run three jobs (one per response model and its verifier):

```bash
cd $SCRATCH/LLM_Hallucination_Measure
source venv_wb/bin/activate

# 1) DeepSeek V3.2 as response, Qwen as verifier
sbatch --export=TARGET_NAME="DeepSeek V3.2 (DeepSeek)",RESPONSE_HF="deepseek-ai/DeepSeek-V3",VERIFIER_HF="Qwen/Qwen3-Next-80B" \
  scripts/slurm_run_wb_probe.sh

# 2) Qwen3 Next 80B as response, DeepSeek as verifier
sbatch --export=TARGET_NAME="Qwen3 Next 80B (OpenRouter)",RESPONSE_HF="Qwen/Qwen3-Next-80B",VERIFIER_HF="deepseek-ai/DeepSeek-V3" \
  scripts/slurm_run_wb_probe.sh

# 3) Llama 4 Maverick as response, DeepSeek as verifier
sbatch --export=TARGET_NAME="Llama 4 Maverick 17B (Groq)",RESPONSE_HF="meta-llama/Llama-4-Maverick-17B",VERIFIER_HF="deepseek-ai/DeepSeek-V3" \
  scripts/slurm_run_wb_probe.sh
```

Add more `sbatch --export=...` lines for the second verifier for each response model if desired.

---

## 7. Run Interactively (Testing)

For short tests (small model or subset), request an interactive node:

```bash
salloc --account=def-YOUR_ACCOUNT --time=1:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:2
# wait until you get a node, then:
cd $SCRATCH/LLM_Hallucination_Measure
source venv_wb/bin/activate
export HF_TOKEN="hf_xxx"

python scripts/run_wb_cross_model_probe_emnlp2025.py \
  --target-model-name "Llama 4 Maverick 17B (Groq)" \
  --response-model-path-or-hf-id meta-llama/Llama-4-Maverick-17B \
  --verifier-model-path-or-hf-id deepseek-ai/DeepSeek-V3 \
  --sequential-encoders --torch-dtype bfloat16 \
  --subset ce
```

---

## 8. Where to Find Results

- **Slurm logs**: `slurm_wb_probe_<JOBID>.out` and `.err` in the directory where you ran `sbatch`.
- **Probe outputs**: under the path printed at the start of the run (default:  
  `data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025/<TargetName>_<VerifierName>/`), or under `$SCRATCH/.../wb_probe_out/` if you use the script’s scratch logic.
- **CSVs**: `*_metrics_by_seed.csv`, `*_metrics_summary.csv`, `*_test_scores.csv`, and `*_run_report.json`.

---

## 9. Cluster-Specific Notes

- **Killarney**: See [Alliance Wiki – Killarney](https://docs.alliancecan.ca/wiki/Killarney) for partitions, module names (`module avail`), and recommended `#SBATCH` options.
- **Vulcan**: Login `vulcan.alliancecan.ca`; access may be limited to Amii-affiliated PIs. Use their wiki for Slurm account and partitions.
- **Account**: Replace `def-<PI_USERNAME>` in the Slurm script with your actual CCDB account (often `rpp-<username>` or similar; check with your PI or Alliance documentation).
