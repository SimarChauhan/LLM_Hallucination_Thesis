# Step-by-Step: Run the White-Box Probe on Nibi

Use **Nibi** (8× H100 per node) with your **def-ilie** account. If SSH closes right after Duo, complete Nibi access in CCDB first (Step 1).

---

## Step 1: Ensure Nibi access (CCDB)

1. Log in to **https://ccdb.alliancecan.ca/**
2. Go to **Resources** → **Access Systems** (or the link from “My Resources and Allocations”).
3. Find **Nibi** and complete any **Request access** or **Accept agreement** steps.
4. Wait a few minutes, then try SSH again.

---

## Step 2: SSH to Nibi

**On your Mac** (Terminal):

```bash
ssh ssimran5@nibi.alliancecan.ca
```

Complete Duo when prompted. You should get a shell like `ssimran5@nibi1:~$`.  
If you see “Connection closed” right after Duo, finish Step 1 and try again (or contact support@tech.alliancecan.ca).

---

## Step 3: Copy the project to Nibi (from your Mac)

**On your Mac** (new Terminal window; do **not** run this inside the SSH session):

```bash
cd /Users/simar/LLM_Hallucination_Measure

rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  . \
  ssimran5@nibi.alliancecan.ca:~/LLM_Hallucination_Measure/
```

Enter your password and complete Duo. Wait until the transfer finishes.

---

## Step 4: On Nibi – go to the project and check data

**In your SSH session on Nibi**:

```bash
cd ~/LLM_Hallucination_Measure
ls data/results/evaluated/results_v2_phase2_eval_no_gemini_4842.final.analysis_ready.skip_greedy_semantic_eval.jsonl
```

If the file is missing, run the rsync from Step 3 again from your Mac.

---

## Step 5: On Nibi – load modules and create venv

```bash
cd ~/LLM_Hallucination_Measure

module avail python
# Pick a version, e.g. module load python/3.11

module load python/3.11
module load gcc
# If pip install later fails on pyarrow:  module load arrow  (then deactivate, load arrow, activate, pip install again)

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch transformers accelerate
pip install pandas matplotlib seaborn pyyaml tqdm python-dotenv
# Skip pyarrow if it fails; or: deactivate, module load arrow, source venv/bin/activate, then pip install -r requirements.txt
```

---

## Step 6: On Nibi – add HuggingFace token

```bash
echo "HF_TOKEN=hf_YOUR_REAL_TOKEN" >> .env
```

Replace with your token from https://huggingface.co/settings/tokens

---

## Step 7: On Nibi – set account in the Slurm script

```bash
nano scripts/slurm_run_wb_probe.sh
```

Ensure this line is set to your Nibi account:

```text
#SBATCH --account=def-ilie
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

---

## Step 8: On Nibi – submit the job

**Option A – Default (Qwen 80B + Llama 400B verifier)**  
The 400B verifier needs ~800 GB; one Nibi node has 8×80 GB = 640 GB, so use a **smaller verifier** (Option B) unless you have multi-node set up.

**Option B – Smaller verifier (fits on one node, recommended)**:

```bash
sbatch -A def-ilie \
  --export=ALL,\
TARGET_NAME='Qwen3 Next 80B (OpenRouter)',\
RESPONSE_HF='Qwen/Qwen3-Next-80B-A3B-Instruct',\
VERIFIER_HF='Qwen/Qwen3-Next-8B',\
BATCH_SIZE=1,SEQUENTIAL_ENCODERS=1,TORCH_DTYPE=bfloat16 \
  scripts/slurm_run_wb_probe.sh
```

**Option C – Same as Option B but different target (e.g. DeepSeek)**:

```bash
sbatch -A def-ilie \
  --export=ALL,\
TARGET_NAME='DeepSeek V3.2 (DeepSeek)',\
RESPONSE_HF='deepseek-ai/DeepSeek-V3',\
VERIFIER_HF='Qwen/Qwen3-Next-80B',\
BATCH_SIZE=1,SEQUENTIAL_ENCODERS=1 \
  scripts/slurm_run_wb_probe.sh
```

You should see: `Submitted batch job 12345` (job ID may differ).

---

## Step 9: On Nibi – check job status and logs

```bash
squeue -u ssimran5
```

When the job is no longer in the list, check the log (replace JOBID with your number):

```bash
cat slurm_wb_probe_JOBID.out
cat slurm_wb_probe_JOBID.err
```

---

## Step 10: Find the results

Outputs are under the project or scratch, for example:

```bash
ls ~/LLM_Hallucination_Measure/data/results/analysis/final_analysis_ready/whitebox/wb_cross_model_probe_emnlp2025/
# or, if $SCRATCH is set:
ls $SCRATCH/LLM_Hallucination_Measure/wb_probe_out/
```

Look for `*_metrics_summary.csv`, `*_test_scores.csv`, and `*_run_report.json`.

---

## Quick checklist

| Step | Where   | Action |
|------|--------|--------|
| 1    | CCDB   | Nibi access / agreements |
| 2    | Mac    | `ssh ssimran5@nibi.alliancecan.ca` |
| 3    | Mac    | `rsync ... ssimran5@nibi.alliancecan.ca:~/LLM_Hallucination_Measure/` |
| 4    | Nibi   | `cd ~/LLM_Hallucination_Measure` and check data file |
| 5    | Nibi   | `module load python/3.11 gcc`, create venv, pip install |
| 6    | Nibi   | Add `HF_TOKEN` to `.env` |
| 7    | Nibi   | Set `#SBATCH --account=def-ilie` in `slurm_run_wb_probe.sh` |
| 8    | Nibi   | `sbatch -A def-ilie --export=ALL,... scripts/slurm_run_wb_probe.sh` (use smaller verifier) |
| 9    | Nibi   | `squeue -u ssimran5`, then `cat slurm_wb_probe_JOBID.out` |
| 10   | Nibi   | Get results from output dir |

---

## If SSH closes right after Duo

- Finish **Step 1** (Access Systems for Nibi in CCDB).
- Try again after a few minutes.
- If it still closes: `ssh -v ssimran5@nibi.alliancecan.ca`, complete Duo, and note the last lines before “Connection closed”; then email **support@tech.alliancecan.ca** with that and your username.
