# Business Entity Resolution: Methodology

## 1. Summary

A multi-stage entity-resolution system:

1. **Multi-view normalization** with alias and token-stability tables learned
   from the training labels (no external data).
2. **Candidate generation**: union of exact multi-view keys, rare-token
   inverted indexes and a **fine-tuned multilingual-E5** retriever
   (name view + name-and-address view), pruned bidirectionally.
   Candidate recall **0.9994** at ~52 candidates per S1 entity.
3. **Two-stage GPU XGBoost pair classifier** (~150 features); stage 2 adds
   target-side competition, exploiting the fact that every matched S2/S3
   record has exactly one true S1 parent.
4. **Cross-encoder reranker** (fine-tuned E5 reading both records jointly)
   applied only to the ~4.5% of pairs the GBDT is uncertain about, merged by a
   stage-3 combiner.
5. **Learned pruner**: the stage-1 GBDT keeps each S1's top-10 candidates per
   source; this is the final candidate set (`candidate_pairs.tsv`, ~20 per
   S1) and the only set the matcher (stage 2 + reranker + stage 3) sees.
6. **Decision layer**: calibrated probability, target-uniqueness filter,
   threshold tuned on macro-F0.5.

**Honest validation macro-F0.5 = 0.9908** on a held-out slice of S1 entities
(fold 4, ~441K entities) that was excluded from *every* learned component: the
encoder fine-tuning, the GBDT folds, the cross-encoder, and the stage-3
combiner (which uses sub-folds inside fold 4).

| Stage | Honest fold-4 macro-F0.5 |
|---|---|
| Perfect classifier on blocked candidates (ceiling) | 0.99983 |
| Stage-2 GBDT, uniqueness filter + threshold | 0.98428 |
| + cross-encoder reranker + stage 3 | 0.99064 |
| + learned pruner to 20 candidates / S1 | 0.99062 |
| **+ second reranker (e5-base) + 3-seed stage 3 (submitted)** | **0.99075** |

France has no labels; its components are described in section 10.

## 2. What the data actually looks like

Measured on the full training set before any modelling:

| Fact | Value | Consequence |
|---|---|---|
| Positive pairs | 7,638,365 | - |
| Country disagreement across positives | **0** | Country is a safe hard partition |
| Targets claimed by more than one S1 | **0 of 7,638,365** | Target side is strictly one-to-one |
| S2 / S3 records that match something | 73.4% / 74.6% | ~26% are distractors |
| Matches per S1 | mean 3.46, max 11 | Top-1 matching is wrong |
| Singleton S1 entities | 5.59% | Worth 5.6 F0.5 points on their own |
| S1 records sharing a normalized name | 39.3% (largest bucket 253) | Name alone cannot identify |
| Non-ASCII names in S2 / S3 | ~15% | Cross-script matching is required |
| S1 records that are non-ASCII | 0% (train) | Matching is always ASCII -> script |

The noise is synthetic and its operator set is visible in the pairs: legal
suffix add/drop, case, accent injection (`Sánderson`), homoglyph typos
(`5anderson`, `Fa1cone`, `8rigitte`), token duplication and transposition,
truncation, `#handle` and `domain.com` forms, `f/k/a <real name>` prefixes,
acronyms (`DS` <- Dream Systems), whole-name replacement (`Ariagild`), generic
token swaps (`Hospital` -> `Center`), and Indic transliteration. Addresses add
component reordering, abbreviation, state aliasing, ordinal<->word
(`7th` <-> `Seventh`), zero padding (`0033`), leading-digit loss (`513` -> `13`),
ranges, fractions, `NULL` literals and injected components (`PMB`, `PO BOX`).

## 3. Normalization

Every transformation is an **additional view**; raw fields are never
overwritten, because each transform destroys evidence another view needs.
Names carry seven views (NFKC, punctuation-folded, compact, homoglyph-folded,
suffix-stripped, content-only, token-sorted, acronym) and addresses nine
(canonical, compact, token-sorted, content-only, number sequence, number set,
first/last number, postal candidate).

Two bugs found and fixed here mattered more than any model tuning:

* **Indic shredding.** Python's `\w` excludes Unicode combining marks, so a
  naive `[^\w\s]` strips Devanagari matras and turns
  `राम मार्केटिंग प्राइवेट लिमिटेड` into fifteen single consonants. This
  silently corrupted 15% of S2/S3 before it was caught by single Devanagari
  characters appearing 273,000 times in the token statistics. Marks are now
  preserved for non-Latin bases and stripped only from Latin ones, where
  accent injection is genuinely noise.
* **Ordinal/zero canonicalization.** `309 8th Street, IA` and
  `309. EIGHTH ST, ATLANTIC, IA` must produce the same token multiset.

## 4. Learned lookup tables (no external data)

No gazetteer, geocoder or external source is used. Street-type and state
aliases, legal suffixes, generic business words and transliteration
equivalences are all **mined from `train_ground_truth.tsv`**: for each matched
pair the two token sets are aligned, and a clean one-for-one residual is a
witnessed substitution. Aggregated over millions of pairs this yields
`rd->road`, `ia->iowa`, `ಕರ್ನಾಟಕ->karnataka`, `महाराष्ट्र->maharashtra`,
`सदर्न->southern`, `ltd->limited`, and the homoglyph family
(`5terling->sterling`, `lnc->inc`).

Two guards were required, both from observed failures:

* **Ambiguous abbreviations.** `st` witnesses both `street` and `saint`; `ct`
  both `court` and `connecticut`. A union-find over witnessed edges fused
  Connecticut with Court. Edges are therefore oriented abbreviation ->
  expansion and never transitively closed, so an ambiguous short form simply
  declines to rewrite.
* **Generic sinks.** The generator swaps a real word for filler
  (`Rns Hospital` -> `Rns Center`), so hundreds of unrelated tokens appear to
  "substitute" for `center`. Unguarded this produced `ford->center` and
  `mr->limited`, which would merge distinct businesses. A target that many
  distinct sources map into is classified as a generic sink and diverted into
  the low-information vocabulary instead of the alias map - which is where it
  belongs, since those tokens must also be weak evidence in features.

Token *stability* is measured the same way: the drop rate of each token across
true matches, computed **after** alias canonicalization (measured before, `rd`
scores 0.887 "unstable" purely because its partner wrote `Road`).

## 5. Candidate generation

Channels, unioned per (country, target-source) partition:

| Channel | Role |
|---|---|
| 11 exact multi-view keys | high-precision anchors |
| rare-token inverted index (name, address) | survives partial corruption |
| address number index | truncated-address slice |
| fine-tuned E5 ANN, combined view (top-20 kept) | the dominant recall channel |
| fine-tuned E5 ANN, name view (top-10 kept) | cross-script names |

The lexical stack alone reached 0.974 recall (India, the harder country) and
its residual was 83% Indic-script names. Adding the fine-tuned encoder made
the two most expensive lexical channels (character-n-gram and
number-by-rare-token indexes) redundant, so they are disabled in the final
("lite") configuration, which is both faster and higher-recall:

| Partition | Candidate recall | Candidates / S1 |
|---|---|---|
| S2 India | 0.99967 | 48.4 |
| S2 US | 0.99943 | 53.7 |
| S3 India | 0.99926 | 48.9 |
| S3 US | 0.99932 | 54.7 |

Pruning is bidirectional (top-k per S1 **or** per target) because the ground
truth is one-to-many from S1 but one-to-one from the target. Three pruning
bugs were found by measurement:

1. Blending raw E5 cosine into the lexical ranking score destroyed it: base-E5
   cosines sit in a 0.90-0.92 band for nearly every pair, so 88% of retained
   candidates were selected essentially at random. Neural pairs now have their
   own retention quota.
2. The global final prune silently undid per-block neural retention
   (recall 0.957 -> 0.971 when fixed).
3. Unweighted containment saturates on short target addresses; the ranking
   score is IDF-weighted over a shared dense vocabulary.

### Learned pruner (last filter before the matcher)

The blocked set (~101 candidates per S1 on test) is narrowed by the stage-1
GBDT to each S1's top-K candidates per source by stage-1 probability, plus any
pair with p1 >= 0.05. The matcher only ever scores the survivors; its
set-level statistics are recomputed over the pruned universe. Measured on the
honest fold-4 slice:

| K per source | Candidates / S1 | True links kept | Honest macro-F0.5 |
|---|---|---|---|
| unpruned | ~104 | 99.94% (blocking) | 0.99064 |
| 6 | 12.3 | 99.955% of blocked | 0.99059 |
| 8 | 16.1 | 99.972% of blocked | 0.99059 |
| **10 (used)** | **20.0** | **99.983% of blocked** | **0.99062** |
| 12 | 24.0 | 99.991% of blocked | 0.99061 |

## 6. Neural retrieval

`intfloat/multilingual-e5-small` (MIT, 118M parameters) fine-tuned **on
challenge pairs only** with in-batch softmax loss, cross-script positives
oversampled to half of each batch, and both a name view and a name+address
view. Fine-tuning excluded fold 4 so its retrieval quality could be measured
honestly. India S2, queries never seen in fine-tuning:

| View | Encoder | recall@10 | recall@30 |
|---|---|---|---|
| combined | base | 0.913 | 0.931 |
| combined | **fine-tuned** | **0.9987** | **0.9997** |
| name | base | 0.479 | 0.556 |
| name | **fine-tuned** | **0.739** | **0.857** |

The held-out and in-sample numbers agree to three decimals, so the gain is
generalisation, not memorisation.

## 7. Pair model

~150 features: multi-view set overlap (Jaccard, Dice, both containments,
cosine) computed by numba kernels over CSR-packed id sets; edit, token-sort,
token-set, partial and Jaro-Winkler similarity via `rapidfuzz.process.cpdist`;
numeric agreement and conflict; retrieval provenance; cross-field interactions;
and the exact fine-tuned-E5 cosine for **every** pair (name and combined views)
with its rank and margin on both the S1 and the target side. That last group
fixed a real defect: the retrieval cosine only existed for pairs the ANN
happened to return, so for every other pair it read 0 - "missing", not
"dissimilar" - and error analysis showed both main error families sitting on
that gap.

Two XGBoost stages on GPU, the five S1-grouped folds trained in parallel, one
per GPU. Stage 2 adds target-side competition over stage-1 out-of-fold scores
(rank among all S1 entities claiming the target, margin to the best claim,
number of strong claims); `t_prank` is the second most important feature
overall. Training uses hard-negative sampling (every positive, each S1's
three most confusable negatives, 3% of the rest); isotonic calibration on the
full out-of-fold set removes the resulting probability bias.

Folds are defined by a stable hash of the S1 `entity_id`, shared by the
encoder fine-tuning, the GBDT and the reranker, so fold 4 is unseen by all.

## 8. Cross-encoder reranker and stage 3

Error analysis of stage 2 on fold 4 showed its remaining mistakes need both
records read jointly: a pseudo-word brand name on the S1's exact address is a
match (the generator's name-replacement operator), while a one-token change at
the same address is a planted distractor; a Devanagari rendering of the S1 name
with a truncated address is a match. The reranker is the fine-tuned E5 encoder
used as a cross-encoder over "name | address" pairs, trained on folds 0-3 pairs
whose stage-2 probability lies in (0.005, 0.995) - about 4.5% of candidates but
nearly all errors - on five GPUs with DDP.

A stage-3 GBDT combines p1, p2, the cross-encoder score and their S1-side and
target-side competition. It is fit with entity sub-folds **inside fold 4**, and
target-side competition is computed over S1 entities of **all** folds, exactly
as at test time. An earlier version that restricted competition to fold-4
pairs was replaced after its in-harness stage-2 baseline disagreed with the
known value; the corrected harness reproduces stage 2 (0.98428 vs 0.98421) and
measures stage 3 at 0.99064.

## 9. Decision layer

For one S1 entity with k predicted and N true matches,
F0.5 = 1.25*TP / (k + 0.25*N) (the TP terms cancel). Calibrated scores pass
through a target-uniqueness filter (each target keeps only its strongest S1
claim), then a threshold tuned on honest macro-F0.5: 0.70 for both India and
US; unseen countries use the global value. An expected-F0.5 set-selection rule
derived from the formula above was also evaluated and scored slightly below
the uniqueness-plus-threshold rule, so the simpler rule is used.

Loss analysis of stage 2 on fold 4: 79% of lost score comes from missed
links, 15% from false merges. About half of the missed links are targets with
no address whose name is shared by several S1 entities; these carry no
distinguishing evidence and abstaining is the F0.5-optimal choice.

## 10. Unseen country (France): self-trained lookup tables

France appears only at test time. No country feature is given to any model;
country is only a partition key, treated as an open set of strings.

The first submission scored 0.983 on the public leaderboard against 0.9906 on
honest validation. Comparing the same model's behaviour on test and on the
training out-of-fold data isolated the gap to France: India and US test pairs
had the same uncertainty profile as training (~12 pairs per 100 S1 with
calibrated probability in 0.3-0.9), while France had twice as many (25.7).
Two causes, both in the lookup tables rather than the models:

* **Borrowed US state aliases.** With no French table, addresses fell back to
  the pooled alias map, whose US state codes rewrote French function words
  (`de` -> `delaware`, `la` -> `louisiana`); `delaware` became the most frequent
  token in French S1 addresses. Unseen countries now receive no borrowed
  aliases at all.
* **Unknown legal forms.** `sarl`, `sas`, `sasu`, `eurl`, `sa`, `sci`, `cie` were
  treated as identifying name tokens, so the generator's suffix swaps
  (`... sci` vs `... sas`) counted as name conflicts.

Fix: the same alias and token-stability miners used on the training labels
are run over high-confidence predicted France pairs (calibrated p >= 0.85,
~858K pseudo-positive pairs). They recover `r -> rue`, `av -> avenue`,
`bd -> boulevard`, `imp -> impasse`, `q -> quai`; French legal forms with drop
rates 0.34-0.52 (the same range as `llc`/`ltd` in US/India data); and French
generic words (club, comite, ecole, amicale, union) as generic sinks.
Pseudo-labels under-count drops, because suffix-swapped pairs are less often
high-confidence, so self-trained cut-offs are 0.05 lower (0.30 / 0.50).
France views, candidates and features are rebuilt with these tables and
re-scored; India and US are untouched, since competition never crosses
countries. This uses only the challenge's unlabelled test records - no
external data.


The mined France tables used for the submitted run are shipped in
`models/` so the run can be reproduced exactly.

### Root cause and fix: a French-adapted cross-encoder

The self-trained tables alone did not move the leaderboard (0.983 -> 0.9826).
A controlled leaderboard decomposition - resubmitting with every French row
left empty - showed India+US at ~0.991 on test (matching the honest 0.9906
validation) and France at ~0.935, so the whole gap was French.

Stage-by-stage diagnostics then located the failing component. The stage-2
GBDT behaves normally on France (its fold models agree; uncertainty close to
training). The stage-3 cross-encoder does not: trained only on US/India text,
on French uncertain-band pairs it was itself unsure 31.6% of the time
(3.6-4.2% elsewhere) and called 8.6% of stage-2-rejected pairs matches
(0.2-0.3% elsewhere). Stage 3 trusts it heavily (97.5% accurate in-domain), so
it overrode correct stage-2 decisions.

The cross-encoder was continued-trained on ~2.5M French pairs with near-certain
stage-2 labels (824K positives: calibrated p >= 0.97 and winning their target;
1.65M hard negatives: highly ranked candidates at p <= 0.02) plus 1M replayed
US/India pairs. On France its unsure rate fell to 2.2% and false "match" calls
to 0.22%, in line with India/US; France's uncertain final decisions halved
(49.1 -> 20.5 per 100 S1). Leaderboard: 0.9826 -> **0.9858**.


### Round 2: removing a bias the first adaptation inherited

Leaderboard A/B tests on the French decision threshold (0.5 < 0.7 < 0.85 <
0.92 < 0.95, each step better) showed French links were still over-accepted.
The remaining false matches were same-address distractors differing only in an
association type word (`traversee comite` vs `traversee ecole`). Two causes:
the self-trained tables had classified those type words as generic filler
(they were mined from pseudo-labels of a model already accepting such pairs),
and the first French cross-encoder had learned the same bias from those
labels. Self-trained generic words are now kept only if they also drop out
across pseudo-matches (drop rate >= 0.30); the type words fail that test and
are name content again (V4). The French cross-encoder was then retrained from
the original US/India reranker on labels from the corrected stage 2 (V6).
Leaderboard: 0.98696 (V3, th 0.95) -> 0.98756 (V4) -> **0.98782 (V6)**.
Removing the cross-encoder for France entirely scored lower (0.98642), so the
adapted reranker is kept.

### Round 3: legal-form decoys and contested records

France predicts the same number of links per S1 as India/US (3.39 vs 3.39,
matching the training distribution bin for bin) yet scores ~0.02 lower, so its
errors are wrong records chosen in place of right ones, not missing or extra
volume. Two patterns explain part of it; both rules are in
`src/run_legalrule.py`, with legal forms parsed by `src/er/legalform.py`.

* **Legal-form decoys (France).** The French generator plants records of a
  sibling company: the same name with a different legal form (sarl/sas/sasu
  -> sa, sas -> sasu, sci -> snc, or sa added to a form-less name) on the same
  street at another house number. The self-trained tables treat French legal
  forms as droppable suffixes and dotted spellings (`s.a.r.l.`, `s.a.s`) shred
  into letters, so the model sees these decoys as perfect matches. Measured on
  France's own reliable links (house number agrees with the S1), the noise in
  true matches only drops a form (30%) or adds a random one (18%, ~3% per
  form); it never replaces one (form conflicts: 0.03% of those links). Among
  links whose house number differs from the S1 and its other matches, 46% have
  a form conflict, and every conflict type carries a different number 97% of
  the time (non-conflict links: 1.3%). A form-less S1 gaining `sa` at another
  number occurs 873 times against ~27 for any other form. The same noise rates
  hold on India/US labels (keep 66%, drop 34%, add 15%), where forms survive
  normalisation and the labelled model already rejects conflicts (conflicting
  candidates are 4.8% true vs 28.9% with equal forms). Rule: drop French links
  with a legal-form conflict, and links where a form-less S1's record gains
  `sa`/`snc` at a different house number (~7,000 links).
* **Contested records (all countries).** Many S1 companies share a name
  across towns; an address-less record matching several cannot be attributed
  from content. On India/US labels (stage-2 OOF, every S1 scored alike) the
  winner of such a contest is right 43% of the time, below the ~73-77%
  precision a link needs under F0.5; a legal-form likelihood (the noise rates
  above) raises that only for posteriors >= 0.9 (77%). French scores stay near
  1 for every contender (0.72% of French links have a runner-up >= 0.9 vs ~0% on
  India/US). Rule: drop a winner whose target has another claimant at cal >= 0.3
  unless its form posterior is >= 0.9, never removing an S1's last link (for a
  one-link entity a ~50% link breaks even). Fold-4 check: +0.00003 (the fold
  holds a fifth of the rivals, so test contests are ~5x denser).

### Second reranker and seed-bagged stage 3 (India/US)

A multilingual-e5-base cross-encoder (MIT, 278M parameters; embeddings frozen,
2M pairs from S1 folds 0-3) was trained as a second reranker for the same
uncertain band. On its own it is no better than the task-tuned e5-small one
(fold-4 band accuracy 0.9746 vs 0.9754), but as an extra stage-3 feature with
three seed-bagged boosters per sub-fold the honest fold-4 score rises from
0.99064 to 0.99075 (thresholds India 0.7, US 0.75). France keeps its
French-adapted pipeline. Per-entity expected-F0.5 set selection was also tested
and scored lower (0.99055), so the global threshold is kept.

Leaderboard: the legal-form and contest rules with the second reranker (V17)
scored **0.989229**, up from 0.98782, which puts France at about 0.977
(India/US ~0.991).

A France-specific GBDT self-trained on pseudo-labels was also tested on an
unseen-country test bed with real labels (US-only model applied to India:
0.96116). Self-training scored 0.95289 and a blend 0.96122 despite 99.6%-clean
pseudo-labels, so it is not used.

### Achievable ceiling

On the honest India/US slice, 25,941 true links have a target with no address
and a rival S1 with an identical core name; no content evidence separates
them (record IDs and file order were checked and carry no signal). Fixing every
other error would reach 0.9946, so the practical overall ceiling on this data
is about 0.993-0.994.

## 11. Compute

7 x RTX 2080 Ti (training capped at 5 so the machine stays usable for others),
40 CPU cores, 440 GB RAM. Heavy intermediates (~150 GB) live on local NVMe.
