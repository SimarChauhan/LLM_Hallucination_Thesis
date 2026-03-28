# Step-by-Step: Run the White-Box Probe on Alliance Canada

Use **Killarney** (`slurm_run_wb_probe.sh`), **Trillium** (`slurm_trillium_wb_probe_replay_matched.sh`), or **Nibi** (`slurm_nibi_wb_probe.sh` + `submit_all_nibi_wb_probes.sh`). Steps 1–3 are the same; step 4 differs by script.

---

## Step 1: Get access and SSH to the cluster

1. Have an active **CCDB account** and know your **account name** (e.g. `def-username` or `rpp-username`). Get it from [CCDB](https://ccdb.alliancecan.ca/) or your PI.
2. For **Killarney**: you may need an **AIP-type RAP** from your AI institution.
3. SSH to the login node:
   - **Killarney:** `ssh YOUR_USERNAME@killarney.alliancecan.ca`
   - **Trillium:** `ssh YOUR_USERNAME@trillium.alliancecan.ca` (or the hostname in Alliance docs)
   - **Nibi:** `ssh YOUR_USERNAME@nibi.alliancecan.ca`

---

## Step 2: Put the project and data on the cluster

**Option A – Clone from Git (if the repo is on GitHub/GitLab)**

```bash
cd $SCRATCH
git clone https://github.com/YOUR_USER/LLM_Hallucination_Measure.git
cd LLM_Hallucination_Measure
```

**Option B – Copy from your laptop with rsync**

On your **laptop** (replace `USER` and `CLUSTER`):

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  /Users/simar/LLM_Hallucination_Measure/ \
  USER@killarney.alliancecan.ca:$SCRATCH/LLM_Hallucination_Measure/
```

Then on the **cluster**:

```bash
cd $SCRATCH/LLM_Hallucination_Measure
```

Ensure the **evaluated JSONL** is there:

```bash
ls data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl
```

If it’s missing, copy it with rsync or scp from your laptop into that path.

---

## Step 3: Set up the environment on the cluster

Run these on the **cluster** (in the repo directory):

```bash
cd $SCRATCH/LLM_Hallucination_Measure

# Load Python (adjust for your cluster; uncomment one)
# module load python/3.11
# module load StdEnv/2023 python/3.11 cuda

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install PyTorch with CUDA (check cluster docs for CUDA version)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
pip install transformers accelerate

# HuggingFace token (required for Llama). Replace with your real token.
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxx" >> .env
# Or: export HF_TOKEN=hf_xxxxxxxxxxxxxxxx  (and add to ~/.bashrc if you want)
```

Check that the probe script runs:

```bash
python scripts/run_wb_cross_model_probe_emnlp2025.py --help
```

---

## Step 4a: Run on Killarney (Performance, 8× H100)

1. **Set your account in the script** (one-time):

   ```bash
   nano scripts/slurm_run_wb_probe.sh
   ```
   Change:
   - `#SBATCH --account=def-<PI_USERNAME>` → `#SBATCH --account=def-YOUR_ACTUAL_ACCOUNT`
   - Optionally `#SBATCH --mail-user=YOUR_EMAIL@university.ca`

2. **Submit the job** (from repo root):

   ```bash
   cd $SCRATCH/LLM_Hallucination_Measure
   source venv/bin/activate

   sbatch -A def-YOUR_ACTUAL_ACCOUNT scripts/slurm_run_wb_probe.sh
   ```

   The script defaults: target **Qwen3 Next 80B (OpenRouter)**, response **Qwen/Qwen3-Next-80B-A3B-Instruct**, verifier **meta-llama/Llama-4-Maverick-17B-128E-Instruct**.  
   **Note:** The 400B verifier needs 800GB; Killarney has 640GB per node. Use a smaller verifier (e.g. dense 17B) or see the “Verifier too large” note in the main docs.

3. **Optional – different target/response/verifier:**

   ```bash
   sbatch -A def-YOUR_ACTUAL_ACCOUNT \
     --export=ALL,\
   TARGET_NAME='DeepSeek V3.2 (DeepSeek)',\
   RESPONSE_HF='deepseek-ai/DeepSeek-V3',\
   VERIFIER_HF='Qwen/Qwen3-Next-80B' \
     scripts/slurm_run_wb_probe.sh
   ```

---

## Step 4b: Run on Trillium (4× H100, replay-matched script)

1. **Set your account** in the script (one-time):

   ```bash
   nano scripts/slurm_trillium_wb_probe_replay_matched.sh
   ```
   Change `#SBATCH --account=def-<PI_USERNAME>` to `#SBATCH --account=def-YOUR_ACTUAL_ACCOUNT`.

2. **Submit with required variables** (Trillium script does **not** set target/response by default; you must pass them):

   ```bash
   cd $SCRATCH/LLM_Hallucination_Measure
   source venv/bin/activate

   sbatch -A def-YOUR_ACTUAL_ACCOUNT \
     --export=ALL,\
   TARGET_MODEL_NAME='Qwen3 Next 80B (OpenRouter)',\
   RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',\
   VERIFIER_HF='meta-llama/Llama-4-Maverick-17B-128E-Instruct',\
   BATCH_SIZE=1,SEQUENTIAL_ENCODERS=1,TORCH_DTYPE=bfloat16 \
     scripts/slurm_trillium_wb_probe_replay_matched.sh
   ```

   **Verifier size:** Llama-4-Maverick-17B-128E is 400B params (~800GB). Trillium has 4×80GB = 320GB per node, so this verifier will not fit. Use a smaller verifier, e.g.:

   ```bash
   VERIFIER_HF='Qwen/Qwen3-Next-8B'   # or another model that fits in 320GB
   ```

3. **Example with a smaller verifier (fits on Trillium):**

   ```bash
   sbatch -A def-YOUR_ACTUAL_ACCOUNT \
     --export=ALL,\
   TARGET_MODEL_NAME='Qwen3 Next 80B (OpenRouter)',\
   RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',\
   VERIFIER_HF='Qwen/Qwen3-Next-8B',\
   BATCH_SIZE=1,SEQUENTIAL_ENCODERS=1,TORCH_DTYPE=bfloat16 \
     scripts/slurm_trillium_wb_probe_replay_matched.sh
   ```

---

## Step 4c: Run on Nibi (all 6 targets, fixed Qwen/Gemma encoders)

This workflow auto-discovers target model names from:

`data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl`

and submits one job per target. It uses fixed encoders for every target:
- response: `Qwen/Qwen3.5-0.8B`
- verifier: `google/gemma-3n-E2B-it`

1. **Dry-run first** (prints 6 `sbatch` commands without submitting):

   ```bash
   cd $SCRATCH/LLM_Hallucination_Measure
   source venv/bin/activate

   DRY_RUN=1 ACCOUNT=def-YOUR_ACTUAL_ACCOUNT \
     bash scripts/submit_all_nibi_wb_probes.sh
   ```

2. **Submit all 6 jobs:**

   ```bash
   ACCOUNT=def-YOUR_ACTUAL_ACCOUNT \
     bash scripts/submit_all_nibi_wb_probes.sh
   ```

3. **Optional single-target submission** (target must exactly match JSON `model` string):

   ```bash
   sbatch -A def-YOUR_ACTUAL_ACCOUNT \
     --export=ALL,TARGET_NAME='Qwen3 Next 80B (OpenRouter)' \
     scripts/slurm_nibi_wb_probe.sh
   ```

Notes:
- This Nibi variant is valid for comparison, but it is **not** strict response=target replay because response/verifier encoders are fixed across all targets.
- The script intentionally fails fast if `Qwen/Qwen3.5-0.8B` is unsupported in your `transformers` version or if Gemma access/token is missing.

---

## Step 5: Check job status and logs

```bash
squeue -u $USER          # list your jobs
squeue -u $USER -l       # more detail

# After the job finishes, view logs (replace JOBID with the number from sbatch output)
cat slurm_wb_probe_JOBID.out    # Killarney script
cat slurm-wb-replay-match-JOBID.out   # Trillium script
cat slurm-wb-probe-nibi-JOBID.out     # Nibi script
tail -f slurm_wb_probe_JOBID.out     # follow live (Killarney)
```

---

## Step 6: Find the results

- **Killarney:** Output directory is under `$PROJECT_ROOT/data/results/analysis/.../wb_cross_model_probe_emnlp2025/` or `$SCRATCH/.../wb_probe_out/` if `$SCRATCH` is set.
- **Trillium:** Under `$OUT_BASE/$RUN_TAG` (script prints it at start), often `$SCRATCH/.../wb_cross_model_probe_emnlp2025_replay_matched/...`.
- **Nibi:** Under `$SCRATCH/.../whitebox/wb_cross_model_probe_emnlp2025_nibi/<target>_gemma3n_e2b/` (or project-local fallback if `$SCRATCH` is unset).

Look for:

- `*_metrics_by_seed.csv`
- `*_metrics_summary.csv`
- `*_test_scores.csv`
- `*_run_report.json`

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | SSH to cluster (Killarney / Trillium / Nibi). |
| 2 | Put repo + JSONL on cluster (git clone or rsync). |
| 3 | Create venv, install torch/transformers/accelerate, set `HF_TOKEN` in `.env`. |
| 4a | Killarney: edit account in `slurm_run_wb_probe.sh`, then `sbatch -A def-ACCT scripts/slurm_run_wb_probe.sh`. |
| 4b | Trillium: edit account, then `sbatch -A def-ACCT --export=ALL,TARGET_MODEL_NAME=...,RESPONSE_HF=...,VERIFIER_HF=... scripts/slurm_trillium_wb_probe_replay_matched.sh`. |
| 4c | Nibi: `DRY_RUN=1 ACCOUNT=def-ACCT bash scripts/submit_all_nibi_wb_probes.sh`, then run again without `DRY_RUN` to submit all 6 targets. |
| 5 | `squeue -u $USER`; check `slurm_*_JOBID.out`. |
| 6 | Results in output dir printed at job start (CSVs and run_report.json). |

Replace `def-YOUR_ACTUAL_ACCOUNT` and model IDs with your real account and chosen models. Use a verifier that fits in the cluster’s GPU memory (e.g. avoid 400B on a 320GB node).
