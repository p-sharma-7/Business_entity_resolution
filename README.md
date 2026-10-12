# Business Entity Resolution: reproducible pipeline

Resolves records from Source 2 and Source 3 against the deduplicated Source 1
reference, producing `matching_results.tsv` and `candidate_pairs.tsv`.

Solution to the Business Entity Resolution Challenge (macro F0.5): for every
Source-1 business, find all matching records in Sources 2 and 3 across India,
the US and France (France appears only in the test set, with no labels).

**Leaderboard: 0.989229.** Held-out validation (India/US): 0.99075.

Pipeline in one line: multi-view normalisation with lookup tables mined from
the training labels -> exact-key, rare-token and fine-tuned multilingual-E5 ANN
blocking -> two-stage XGBoost pair model -> cross-encoder rerankers on the
uncertain band -> stage-3 combiner -> target-uniqueness decision, plus
label-free French adaptation (self-trained tables, French-adapted reranker,
legal-form decoy and contested-record rules). Full write-up: `METHODOLOGY.md`.

## Model weights and output files

Trained weights, lookup tables and the two large output files are on Hugging
Face: **[pushkarsharma/business-entity-resolution](https://huggingface.co/pushkarsharma/business-entity-resolution)**
(3.3 GB; the model card lists which pipeline step uses each file).

```bash
pip install huggingface_hub
huggingface-cli download pushkarsharma/business-entity-resolution --local-dir models
mkdir -p output && mv models/outputs/*.tsv output/      # matching_results.tsv, candidate_pairs.tsv
```

With the weights in `models/`, training steps can be skipped and test scoring
run directly. The competition data is not included; place it in
`dataset/{train,test}/`. The self-trained France lookup tables
(`models/*_France.json`) are also kept in this repository because they cannot
be re-mined exactly.

Base models: `intfloat/multilingual-e5-small` and `intfloat/multilingual-e5-base`
(MIT). This project is released under the MIT License (`LICENSE`).

## Environment

```bash
conda create -y -n ml_er python=3.11
conda activate ml_er
pip install -r requirements.txt
```

GPU is used for the E5 encoder, FAISS search and encoder fine-tuning. The
pipeline runs without a GPU if `faiss-cpu` replaces `faiss-gpu-cu12`, but the
neural retrieval stage becomes the dominant cost.

## Paths

The project root defaults to the repository root.
Override with environment variables:

```bash
export ER_DATA=/path/to/dataset      # contains train/ and test/
export ER_WORK=/path/to/fast/scratch # ~200 GB of intermediates
export ER_ROOT=/path/to/project      # models/, output/, logs/, eda/
```

The `src/*.sh` files are the tmux/conda wrappers used to run the stages on the
original machine; they are not needed. The Python commands below are the
canonical pipeline.

## Results

Honest macro-F0.5 on a held-out 20% of S1 entities (fold 4), excluded from
every learned component (encoder fine-tuning, GBDT folds, cross-encoder,
stage-3 combiner):

| Stage | macro-F0.5 |
|---|---|
| ceiling (perfect classifier on our candidates) | 0.99983 |
| stage-2 GPU XGBoost | 0.98428 |
| + cross-encoder reranker + stage 3 | 0.99064 |
| + learned pruner to 20 candidates / S1 | 0.99062 |
| + second reranker (multilingual-e5-base) and 3-seed stage 3 (submitted) | **0.99075** |

Blocking recall on train: 0.9994 at ~104 candidates per S1. The stage-1 GBDT then
prunes to each S1's top-10 per source (~20 per S1, 99.98% of blocked true links
kept); that pruned set is `candidate_pairs.tsv` and is all the matcher sees
(`PRUNE_K` in `run_final.py`, default 10).

## Layout

```
src/er/                 library
  io_utils.py           strict TSV readers (tab-split, no quoting)
  normalize.py          multi-view name/address normalization
  mine_alias.py         learns token aliases from ground-truth pairs
  blocking.py           numpy hash-join exact-key blocking
  rareblock.py          rare-token / rare-n-gram inverted indexes
  partindex.py          per-partition target index
  candgen.py            channel definitions, cheap scorer, pruning
  setfeat.py            numba CSR set-similarity kernels
  features.py           pair feature extraction
  decide.py             expected-F0.5 decision layer
  legalform.py          legal-form parsing (French and India/US forms)
src/run_*.py            stage drivers
models/*_France.json   self-trained France lookup tables used for the submitted run
utils/validate_submission.py
```

## Run order

All paths are configured in `src/er/io_utils.py` (`DATA`, `WORK`, `ROOT`).
`WORK` should point at fast local storage with ~200 GB free. GPU stages use
GPUs 0-2 by default (`ER_GPUS` for `run_train_gpu.py`; `GPUS` in `run_ce.py`
and `run_predict_par.py`); `torchrun --nproc_per_node` sets the GPUs used for
reranker training. Run the commands from the repository root.

```bash
P=python; S=src

# 1. learn alias + token-stability tables from the training labels
$P $S/run_mine_alias.py && $P $S/run_rebuild_alias.py && $P $S/run_mine_tokens.py

# 2. normalization views for all six files
for split in train test; do for s in 1 2 3; do $P $S/run_build_views.py $split $s; done; done

# 3. fine-tune the encoder on train pairs, S1 fold 4 held out
$P $S/run_finetune_e5.py --gpu 0 --holdout_fold 4 --out models/e5_ho
for split in train test; do for s in 1 2 3; do for v in combined name; do
  $P $S/run_encode_e5.py $split $s $v --bs 512 --maxlen 64 --model models/e5_ho; done; done; done
for split in train test; do
  $P $S/run_ann.py $split combined --k 30; $P $S/run_ann.py $split name --k 30; done

# 4. candidates + features
for split in train test; do
  $P $S/run_candgen2.py $split --channels lite --ann_keep 20 --ann_keep_name 10
  $P $S/run_features.py $split; done

# 5. two-stage GPU pair model, stage-2 decision tuning
$P $S/run_train_gpu.py --rounds 5000
$P $S/run_decide.py

# 6. cross-encoder reranker + stage 3 (fit inside held-out fold 4)
$P $S/run_ce.py prep
torchrun --standalone --nproc_per_node=5 $S/run_ce.py train --bs 64 --max_pairs 6000000
$P $S/run_ce.py score train
$P $S/run_stage3.py

# 7. test scoring and final submission
$P $S/run_predict_par.py
$P $S/run_final.py          # writes output/matching_results.tsv, output/candidate_pairs.tsv
$P utils/validate_submission.py --matching output/matching_results.tsv \
   --candidate output/candidate_pairs.tsv --test-dir dataset/test

# 8. France (no labels): French-adapted reranker, trained from the step-6
#    reranker on confident French stage-2 pseudo-labels plus replayed US/India
#    pairs; India/US reranker scores are reused, French pairs re-scored
export ER_CE_FR_TAG=2
$P $S/run_ce_fr.py prep
torchrun --standalone --nproc_per_node=3 $S/run_ce_fr.py train --bs 96 --lr 2e-5
$P $S/run_ce_splice.py
ER_TH_FRANCE=0.97 PRUNE_K=10 $P $S/run_final.py
unset ER_CE_FR_TAG

# 9. second reranker (multilingual-e5-base, India/US), 3-seed stage 3
export ER_CE_BASE=intfloat/multilingual-e5-base ER_CE_OUT=models/ce_b ER_CE_TAG=_b ER_CE_FREEZE_EMB=1 ER_CE_GC=0
torchrun --standalone --nproc_per_node=3 $S/run_ce.py train --bs 32 --lr 3e-5 --max_pairs 2000000
$P $S/run_ce.py score train
ER_S3_SEEDS=3 ER_S3_OUT=models/stage3_ce2_s3 $P $S/run_stage3_ce2.py
$P $S/run_final_ce2.py prep && $P $S/run_ce.py score test
ER_S3_OUT=models/stage3_ce2_s3 $P $S/run_final_ce2.py assemble
unset ER_CE_BASE ER_CE_OUT ER_CE_TAG ER_CE_FREEZE_EMB ER_CE_GC

# 10. decision rules: French legal-form decoys; contested records (all countries)
ER_FINAL=test_final_ce2 ER_TH_FRANCE=0.97 ER_TH_INDIA=0.7 ER_TH_US=0.75 \
  ER_CONTEST=0.3 ER_CONTEST_IU=1 $P $S/run_legalrule.py final
cp output_variants/final/matching_results.tsv output/matching_results.tsv
```

France tables. France has no training labels, so its alias and
token-stability tables (`models/*_France.json`) are mined by
`src/run_mine_selftrain.py` from confident French predictions of a first pass.
The tables used for the submitted run are included in `models/` and step 2
picks them up, which reproduces the submitted files. To regenerate them
instead: delete them, run steps 1-7, run `$P $S/run_mine_selftrain.py`, rerun
step 2 for `test`, rebuild the French candidates and features
(`run_candgen2.py test --only S2_France ...` / `--only S3_France ...` with the
step-4 options, then `run_features.py test`), re-score France with
`ER_ONLY_COUNTRY=France $P $S/run_predict_par.py`, and continue at step 8.
Results can then differ slightly from the submitted files: the first pass that
produced the shipped tables still used an address-alias fallback that has
since been removed.

## Notes

* No external data, gazetteer, geocoder or API is used. Every lookup table
  (street-type and state aliases, legal suffixes, generic business words,
  transliteration equivalences) is mined from `train_ground_truth.tsv`.
* Country is treated as an open set of string labels; France appears only at
  test time and flows through the same code path.
* Models: `intfloat/multilingual-e5-small` (MIT, 118M parameters; retrieval
  encoder and first reranker), `intfloat/multilingual-e5-base` (MIT, 278M;
  second reranker) and XGBoost, well inside the licence and 8B-parameter limits.
