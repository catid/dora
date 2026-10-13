# Initialization study: partial results

The full study is in progress. This export contains the completed SciFact centered, intentional warm-start and separate LoRA-Null/GME comparisons, the repeated centered NFCorpus, warm-start and LoRA-Null/GME comparisons with their SciFact transfer evaluations, the Qwen/ViGGO warm-start and LoRA-Null/GME comparisons, the Aircraft warm-start and extension comparisons, and dense-NF4 retrieval comparisons. The global centered configuration was selected using validation; no test result selects an overall initializer winner. Remaining tasks and the five-fresh-seed confirmation are pending. Confirmation seeds 45–49 cover the main centered grid's validation-selected candidate and its accepted/Gaussian controls; the separate warm-start and LoRA-Null/GME comparisons use seeds 42–44.

The original centered NFCorpus run failed historical-baseline parity. Its complete separate repetition now passes both strict audits, and the repeated cohort's final comparison appears below. The original failure remains unresolved and preserved. Quantized Qwen's corrected final audit and scoring review also passed. See the [recovery note](recovery_note.md) for evidence and reporting limits.

## Centered SciFact

All 602 validation pilots across 76 families completed before test scoring. The exact accepted anchor searches two learning rates; each other family searches four initializer settings × two rates. The global validation choice therefore has a larger search opportunity than either individual control family. All three rows below selected LR `0.001`.

| Initialization | Test nDCG@10 ×100, mean ± sample SD |
|---|---:|
| Accepted DoRA | 67.99 ± 0.64 |
| Scale-tuned original Gaussian (2×) | 69.01 ± 0.62 |
| Centered Gaussian, both factors (validation-selected) | 71.97 ± 0.78 |

![Centered SciFact comparison](centered_scifact_summary.png)

The selected `inilora_alpha__i3__lr1` draws both factors independently from N(0,1) and uses `W + (BA − B₀A₀)`. This tuned IniLoRA-alpha derivative preserves the starting function and uses no optimization-based prefit. Original column normalization, full denominator gradients, dense projection and FP32 magnitudes remain fixed.

Its paired difference versus the accepted anchor is +3.98 points, exploratory 95% interval [1.64, 6.38]; versus the equally tuned Gaussian family it is +2.95 [0.82, 5.09]. Intervals use 10,000 crossed paired seed/query bootstrap draws without multiple-comparison correction. Primary seeds are 42–44; seed 42 reuses its full-budget selected pilot fit. Confirmation seeds 45–49 are still pending and are not pooled into these results.

The final audit passed 754 unique fits (602 pilots and 152 additional training fits), 86,710 optimizer updates and 129,474 initial/validation/test ranking records. This export rescored all 68,400 test rankings from 228 unique selected fits. Both selection lanes share the same selected configurations on this task. [Summary and selected settings](centered_scifact_summary.json), [all 152 family/lane rows](centered_scifact_all_families.csv), [paired comparisons](centered_scifact_comparisons.csv), and [search budgets](centered_scifact_search_budgets.json) retain all families. This result establishes no cross-task winner.

## Centered NFCorpus and transfer: complete repeated cohort

The validation-selected Student-t(df=3) up initializer has no demonstrated improvement in this three-seed comparison. All three displayed configurations use LR `0.0001`, selected using NFCorpus validation; transfer uses the same trained checkpoints. Values are nDCG@10 ×100, mean ± sample SD over seeds 42–44.

| Initialization | NFCorpus | SciFact transfer |
|---|---:|---:|
| Accepted DoRA | 32.02 ± 0.24 | 63.53 ± 0.24 |
| Gaussian scale 2 | 31.82 ± 0.27 | 63.02 ± 0.91 |
| Student-t(df=3) up (validation selected) | 32.09 ± 0.20 | 64.64 ± 0.84 |

![Centered NFCorpus and transfer](centered_nfcorpus/centered_nfcorpus_transfer.png)

The Student-t difference versus accepted DoRA is +0.07 [−0.74, +0.84] on NFCorpus and +1.12 [−0.45, +2.81] on transfer. These exploratory 95% intervals use 10,000 crossed paired seed/query resamples without multiple-comparison correction. Five fresh confirmation seeds remain pending. The [complete task report](centered_nfcorpus/README.md) retains all 76 families, 304 family/lane/dataset rows, exact settings, costs and portable scoring replay.

All 1,204 registered pilots passed the unchanged strict audit before final scoring. Rank-8 MiniLM uses the deduplicated official 2,582/323 train/validation queries, all 323 official NFCorpus test queries, all 300 official SciFact transfer queries and 405 updates per fit. The 357 selected checkpoints include 119 seed-42 pilot reuses and 238 new fits. Independent report review rescored 222,411 final rankings; parts-only portable replay reproduced all 16 report files. The original failed cohort remains unexplained and preserved; these results validate only the complete repeated cohort.

## Separate LoRA-Null/GME extension: direct SciFact

The [extension report](extension_scifact/README.md) retains all six families, validation-selected settings, paired intervals, recorded costs and a portable saved-prediction replay. It completed 42 pilots and 12 additional seed-43/44 fits; 18 selected final fits supply 5,400 test-query rankings. All six families selected LR `0.001` in both selection lanes.

| Extension initialization | Test nDCG@10 ×100, mean ± sample SD |
|---|---:|
| Accepted DoRA | 67.99 ± 0.64 |
| Scale-tuned original Gaussian | 69.01 ± 0.62 |
| LoRA-Null, task-only centered adaptation | 66.76 ± 0.51 |
| GME paper recipe, warm start | 66.20 ± 0.41 |
| GME author V1 recipe, warm start | 66.26 ± 0.18 |
| GME author V2 recipe, warm start | 66.05 ± 0.42 |

Each literature family and Gaussian control searches four settings × two rates; the accepted anchor searches only the two rates. The GME parent receives 24 attempts across three recipes. Null centers its initial product and uses only task-training activation statistics, so this is not a knowledge-retention evaluation. GME preserves each source recipe's intentional nonzero warm start, with original magnitudes and the accepted column DoRA geometry. These lower observed means do not establish performance on the remaining tasks, and the extension does not revise the main grid's validation selection. Three-seed paired intervals are exploratory without multiple-comparison correction. The portable archive verifies scoring and report reproduction, with tensor payloads explicitly omitted and retained locally.

## Separate LoRA-Null/GME extension: NFCorpus and transfer

The [NFCorpus extension report](extension_nfcorpus/README.md) includes all six families and both fixed and tuned selection lanes. These results use NFCorpus validation to select settings and learning rates, then evaluate the same trained checkpoints on both datasets. Values are test nDCG@10 ×100, mean ± sample SD over seeds 42–44.

| Extension initialization | NFCorpus | SciFact transfer |
|---|---:|---:|
| Accepted DoRA | 32.02 ± 0.24 | 63.53 ± 0.24 |
| Scale-tuned original Gaussian | 31.82 ± 0.27 | 63.02 ± 0.91 |
| LoRA-Null, task-only centered adaptation | 31.70 ± 0.09 | 59.77 ± 0.41 |
| GME paper recipe, warm start | 32.19 ± 0.23 | 61.97 ± 0.95 |
| GME author V1 recipe, warm start | 32.03 ± 0.28 | 61.34 ± 0.58 |
| GME author V2 recipe, warm start | 32.02 ± 0.18 | 61.51 ± 0.41 |

![NFCorpus extension and SciFact transfer, both selection lanes](extension_nfcorpus/extension_nfcorpus_transfer.png)

None of the NFCorpus differences versus the tuned accepted anchor has a paired interval excluding zero. On transfer, Null, GME V1 and GME V2 have lower means with exploratory intervals below zero; the GME paper interval crosses zero. For the validation-tuned selections, all four same-learning-rate anchor comparisons cross zero on both datasets. The report retains the tuned anchor, same-learning-rate controls and equally tuned Gaussian comparisons so initialization and selected learning rate are not conflated. Intervals use 10,000 paired seed/query resamples, without multiple-comparison correction.

All 84 pilots completed and passed audit before final test evaluation. The accepted anchor searches four learning rates; each four-setting family searches 16 candidates. The GME parent receives 48 attempts across its three distinct recipes. NFCorpus uses 2,582 training / 323 validation queries after exact-text deduplication of the official 2,590 / 324, all 323 official test queries, rank 8 and 405 updates per fit. Transfer uses all 300 official SciFact test queries without additional training or selection.

Six selected final GME fits (paper and V1, three seeds each) have `b_norm_ratio_ok=false`: the maximum up-factor/base-weight Frobenius norm ratio is approximately 0.00190–0.00205, above the fixed 0.001 small-factor threshold. This flag does not measure whether target moments were equalized. The fits remain finite and pass checkpoint replay. This diagnostic was not a registered numerical-failure gate, so the flagged fits remain in the comparison. These are source-specific initialization adaptations to accepted DoRA, with Null centered and the three GME recipes intentionally uncentered.

The task audit passed 106 unique fits, 42,930 updates and 74,177 initial/validation/test ranking records. The export rescored 20,559 test rankings from 33 checkpoints: 30 family-selected fits plus three same-learning-rate accepted controls. Seed-42 pilots are reused; 22 new fits cover seeds 43–44. All three controls at the additional learning rate remain explicit in cost and scoring evidence. The full report, costs, portable replay and independent review are linked in the task report. All four extension tasks are complete; this extension is outside the main grid's fresh-seed confirmation.

## Separate LoRA-Null/GME extension: Qwen/ViGGO

The [Qwen extension report](extension_extraction/README.md) includes all six families, both selection lanes, same-learning-rate controls, supplementary extraction metrics, costs and portable scoring replay. Values are exact JSON match %, mean ± sample SD across seeds 42–44. Exact match compares schema-valid parsed objects, ignoring JSON key order and whitespace.

| Extension initialization | Fixed LR | Validation-tuned LR |
|---|---:|---:|
| Accepted DoRA | 73.83 ± 2.95 | 73.83 ± 2.95 |
| Scale-tuned original Gaussian | 73.83 ± 2.95 | 67.45 ± 6.25 |
| LoRA-Null, task-only centered adaptation | 70.57 ± 1.80 | 70.57 ± 1.80 |
| GME paper recipe, warm start | 69.92 ± 1.70 | 69.92 ± 1.70 |
| GME author V1 recipe, warm start | 65.10 ± 2.54 | 65.10 ± 2.54 |
| GME author V2 recipe, warm start | 70.96 ± 5.80 | 70.96 ± 5.80 |

![Qwen/ViGGO LoRA-Null and GME comparison, both selection lanes](extension_extraction/extension_extraction.png)

Rank-8 Qwen2.5-3B-Instruct uses deterministic 1,024/128/256 subsets of the official ViGGO splits after seven exact-text training duplicates were excluded. This does not evaluate the full official test split. All 42 pilots trained for 48 updates. Validation NLL selected settings and learning rates before test evaluation. Each four-setting family searches eight candidates; the accepted anchor searches two rates, and the three GME recipes receive 24 attempts in total. The fixed rate is `0.0003`. Only the Gaussian family changes selection between lanes: scale 1 at fixed LR versus scale 2 and LR `0.0001` when tuned. Every other family selects LR `0.0003` in both lanes.

No treatment demonstrates an improvement over accepted DoRA. GME V1 is lower by 8.72 points, with an exploratory paired 95% interval [−16.02, −1.04]. The other validation-tuned treatment comparisons versus accepted DoRA have intervals containing zero, as do all treatment comparisons against the validation-tuned Gaussian. Intervals use 10,000 paired seed/example resamples without multiple-comparison correction. The report retains three extra accepted-anchor fits at LR `0.0001` to separate same-rate comparisons from the tuned-anchor comparison.

Six selected final GME fits, paper and V1 across three seeds, exceed the fixed 0.001 small-factor norm-ratio diagnostic: their observed maxima are approximately 0.00295–0.00342. They remain finite and pass checkpoint replay. This flag does not measure target-moment matching and was not a registered failure exclusion; the report retains these fits and their diagnostics. Null uses task-only activation statistics and does not establish pretraining-knowledge retention. The GME recipes preserve their source-specific nonzero warm starts.

The task audit passed 68 fits including two short controls, 8,170 updates, 14,688 checkpoint tensors and 6,144 saved test predictions. Four historical accepted fits match bitwise. The 42 pilots and 24 fresh final fits account for 66 training fits and 8,160 updates; the two controls add ten updates. Seven family-selected configurations supply 21 final fits, plus three same-learning-rate anchor controls. All finals train for 256 updates, including seed 42; pilot training is not reused. Shared fixed/tuned choices are not independent replicates. Portable replay reproduces saved-prediction scoring and reports; test NLL remains from original inference. This extension does not change main-grid selection and is outside its fresh-seed confirmation.

## Intentional warm starts

All 42 registered validation pilots completed before test scoring; 18 selected full-budget final fits cover six families and three seeds. Seed 42 uses its full-budget pilot; 12 additional fits with seeds 43 and 44 were trained. Selected settings are retained in `scifact_warmstart_summary.json`.

Warm starts intentionally retain their nonzero initial factor product. The accepted baseline has a zero product. Rank-8 MiniLM uses 726 training / 81 validation / 300 official-test queries and 115 optimizer updates per fit. The accepted one-setting anchor searches two learning rates; each treatment searches four initializer settings × two rates. These are unequal initializer search budgets, explicitly retained.

CNT abbreviates [ConsNoTrainLoRA](https://arxiv.org/abs/2507.08044). It computes the adapter initialization from pretrained weights and training activations; ordinary fine-tuning follows. Cross, self and shift refer to its three constraint formulas, adapted here to accepted DoRA.

| Initialization | Test nDCG@10 ×100, mean ± sample SD |
|---|---:|
| Accepted DoRA | 67.99 ± 0.64 |
| LoRA-One warm start | 66.39 ± 0.26 |
| LoRA-DA warm start | 65.59 ± 0.73 |
| CNT cross warm start | 65.91 ± 0.10 |
| CNT self warm start | 41.09 ± 0.40 |
| CNT shift warm start | 65.47 ± 0.32 |

![SciFact intentional warm starts](scifact_warmstart.png)

All 5,400 saved final query rankings were independently rescored again for this export. The copied full task audit checks 54 unique fits (42 pilots plus 12 new final fits), 6,210 optimizer updates and 14,148 initial/validation/test ranking records. It also verifies four historical accepted fits. Raw checkpoints, logs, source and predictions remain under the recorded local raw root; their hashes are preserved. This partial export is not a portable checkpoint archive.

## NFCorpus warm starts and SciFact transfer

The [completed NFCorpus warm-start report](warm_nfcorpus/README.md) retains all six families in both fixed and validation-tuned learning-rate lanes, with a bar chart, paired intervals, costs and portable saved-prediction replay. The following table uses settings selected on NFCorpus validation only. Transfer evaluates those same checkpoints on SciFact, without additional training or tuning.

| Initialization | NFCorpus nDCG@10 ×100 | SciFact transfer nDCG@10 ×100 |
|---|---:|---:|
| Accepted DoRA | 32.02 ± 0.24 | 63.53 ± 0.24 |
| LoRA-One warm start | 32.15 ± 0.02 | 62.16 ± 0.58 |
| LoRA-DA warm start | 32.17 ± 0.13 | 61.68 ± 0.34 |
| CNT cross warm start | 26.70 ± 0.23 | 50.97 ± 0.26 |
| CNT self warm start | 16.97 ± 0.11 | 30.48 ± 0.81 |
| CNT shift warm start | 23.25 ± 0.66 | 38.03 ± 1.12 |

![NFCorpus warm starts and SciFact transfer, both selection lanes](warm_nfcorpus/warm_nfcorpus_transfer.png)

Values are mean ± sample SD across seeds 42–44. Rank-8 MiniLM uses 2,582 training / 323 validation queries after recorded exact-text deduplication, all 323 official NFCorpus test queries, and 405 updates per fit. The official training / validation counts before deduplication are 2,590 / 324; transfer uses all 300 official SciFact test queries. All 84 pilots completed before test scoring. Each treatment searches four initializer settings × four learning rates, while the exact accepted anchor searches only the four rates. The LoRA-One and LoRA-DA differences versus accepted DoRA have exploratory paired intervals crossing zero on both evaluations; all three CNT warm-start intervals are below zero. These intervals have no multiple-comparison correction.

The audit passed 106 unique fits and 42,930 optimizer updates. Eleven distinct selected configurations yield 33 final checkpoints: selected seed-42 pilots are reused and 22 new fits cover seeds 43–44. The report rescored all 20,559 saved test rankings across the two datasets. Transfer adds evaluation work but no training fits. The five-fresh-seed confirmation covers the main centered grid, not this separate warm-start comparison.

## Qwen/ViGGO warm starts

The [completed Qwen warm-start report](warm_extraction/README.md) retains all six families in both selection lanes, with supplementary valid-JSON, schema and slot metrics, paired intervals, costs and portable prediction replay. Values below are exact JSON match %, mean ± sample SD across seeds 42–44. Exact match compares schema-valid parsed objects, so key order and whitespace do not matter.

| Initialization | ViGGO exact match % |
|---|---:|
| Accepted DoRA | 73.83 ± 2.95 |
| LoRA-One warm start | 71.48 ± 3.12 |
| LoRA-DA warm start | 61.85 ± 2.54 |
| CNT cross warm start | 74.48 ± 3.32 |
| CNT self warm start | 0.00 ± 0.00 |
| CNT shift warm start | 74.09 ± 5.19 |

![Qwen/ViGGO warm-start comparison, both selection lanes](warm_extraction/warm_extraction.png)

Qwen2.5-3B-Instruct rank 8 uses deterministic 1,024/128/256 train/validation/test subsets from the official ViGGO splits, after seven exact-text training duplicates were excluded. This does not evaluate the full official test split. All 42 seed-42 pilots used 48 updates. Validation NLL selected each initializer and rate before test evaluation; the five treatments search eight candidates each and the accepted anchor searches two rates. All families selected LR `0.0003` in both lanes. Eighteen fresh final fits use 256 updates each; seed-42 pilot training is not reused, while identical fixed/tuned choices share final checkpoints.

LoRA-DA's difference versus the accepted anchor is −11.98 points, exploratory 95% interval [−18.23, −5.73]. CNT self is also lower and has zero exact matches across all three seeds. Its saved predictions are retained: valid JSON averages 90.23%, schema validity 78.91% and slot F1 11.74%. LoRA-One, CNT cross and CNT shift have paired intervals crossing zero. These source-specific warm-start adaptations show no demonstrated improvement here; the unresolved CNT source-sign boundary remains documented. Intervals use 10,000 paired seed/example draws without multiple-comparison correction.

The task audit passed 62 fits including two controls, 6,634 logged updates, 13,392 checkpoint tensors and all 4,608 final predictions. Four historical accepted fits match bitwise. The report and cost files distinguish the 60 pilot/final training fits from the two controls, and retain missing timers rather than claiming end-to-end wall time. Portable replay reproduces scoring and all report outputs; test NLL remains from original inference and is not independently re-inferred. This substudy is outside the main grid's five-fresh-seed confirmation.

The remaining main-grid results, quantized Aircraft and fresh-seed confirmation are pending. The public implementation and prior study sections remain unchanged. Raw checkpoint tensors remain in the recorded local task roots; portable exports reproduce scoring and reports, not checkpoint inference. Recorded receipt intervals may include scheduling gaps and must not be interpreted as training or total study wall time.

### Completed Aircraft warm starts and extension (2026-10-11)

Several initializations help Aircraft in these separate three-seed studies. LoRA-One and LoRA-DA improve on the accepted anchor by 4.69 and 4.24 points. GME paper scores 61.18% versus 56.27% for accepted DoRA and 56.76% for scale-tuned Gaussian. Its exploratory paired differences are +4.91 [3.24, 6.55] and +4.42 [3.28, 5.60] points, respectively. The main centered search and its fresh-seed confirmation remain pending; this is no cross-task winner claim.

| Initialization | Aircraft macro accuracy %, mean ± sample SD |
|---|---:|
| Accepted DoRA | 56.27 ± 0.74 |
| Scale-tuned Gaussian | 56.76 ± 0.27 |
| LoRA-One warm start | 60.96 ± 1.06 |
| LoRA-DA warm start | 60.50 ± 0.64 |
| CNT cross warm start | 41.41 ± 2.50 |
| CNT self warm start | 10.95 ± 0.66 |
| CNT shift warm start | 33.15 ± 0.90 |
| LoRA-Null, task-only centered | 60.20 ± 0.43 |
| GME paper warm start | 61.18 ± 0.55 |
| GME author V1 warm start | 60.39 ± 0.49 |
| GME author V2 warm start | 60.82 ± 0.04 |

![Aircraft extension results, mean ± sample SD](extension_aircraft.png)

These rank-2 ViT-B/16 runs use all official Aircraft splits (3,334 train / 3,333 validation / 3,333 test). Each task completed 84 ten-epoch pilots before validation-only selection and 18 fresh twenty-epoch final fits at seeds 42–44. All selected families use LR 0.001, so fixed and tuned lanes share checkpoints. Each non-anchor family searches four initializer settings × four rates, versus four rates for the anchor; GME has three separately reported recipes. Intervals use 10,000 paired seed/example resamples with class stratification, without multiplicity correction. Null uses a centered task-only update; the warm starts intentionally change the initial function. These adaptations receive no extra confirmation seeds.

[Aircraft warm-start evidence](warm_aircraft/README.md) and [Aircraft extension evidence](extension_aircraft/README.md) include both selection lanes, costs, raw-prediction reviews and portable report replay. They independently rescore 59,994 final predictions per task. The accepted public implementation remains unchanged.

### Completed dense-NF4 retrieval comparisons (2026-10-11)

Values are test nDCG@10 ×100 over seeds 42–44, with validation-selected settings and learning rates. SciFact transfer uses the NFCorpus selections and checkpoints. Original/quantized names the magnitude initialization source; i1/i3 denotes one/three LoftQ iterations. Each NF4 row is the quantizer-only control paired with the corresponding LoftQ base.

| Recipe | SciFact direct | NFCorpus | SciFact transfer |
|---|---:|---:|---:|
| Accepted DoRA | 67.99 ± 0.64 | 32.02 ± 0.24 | 63.53 ± 0.24 |
| NF4 i1, original magnitude | 67.90 ± 0.55 | 32.19 ± 0.24 | 62.79 ± 0.68 |
| LoftQ i1, original magnitude | 66.69 ± 0.37 | 31.93 ± 0.09 | 61.91 ± 0.38 |
| NF4 i1, quantized magnitude | 67.95 ± 0.57 | 32.19 ± 0.33 | 62.82 ± 0.78 |
| LoftQ i1, quantized magnitude | 66.61 ± 0.70 | 31.94 ± 0.11 | 61.98 ± 0.35 |
| NF4 i3, original magnitude | 68.43 ± 0.46 | 31.94 ± 0.23 | 63.42 ± 0.35 |
| LoftQ i3, original magnitude | 67.41 ± 0.30 | 31.77 ± 0.11 | 62.56 ± 0.41 |
| NF4 i3, quantized magnitude | 68.53 ± 0.60 | 31.98 ± 0.22 | 63.30 ± 0.47 |
| LoftQ i3, quantized magnitude | 67.49 ± 0.60 | 31.66 ± 0.10 | 62.87 ± 0.11 |

![Dense-NF4 NFCorpus comparison, fixed and tuned lanes](quantized_nfcorpus.png)

These dense dequantized simulations do not measure compressed serving. The [direct SciFact report](quantized_scifact/README.md) and [NFCorpus/transfer report](quantized_nfcorpus/README.md) retain same-base/same-rate controls and paired intervals; comparing tuned means alone does not isolate initialization when selected rates differ. Reviewed scoring replay covers 8,100 and 31,773 final rankings, respectively. Each evaluation uses its full official test split; NFCorpus training/validation excludes exact-text duplicates. Seed-42 full-budget pilots are reused and seeds 43–44 are freshly trained.

The remaining centered NFCorpus/Qwen/Aircraft results, quantized Aircraft results and main-only fresh-seed confirmation are pending. All four task exports above include hash-bound evidence and portable scoring replay; checkpoint tensors remain in the local task roots.

### Completed dense-NF4 Qwen/ViGGO comparison (2026-10-11)

None of the four LoftQ variants demonstrates an improvement over accepted DoRA in this three-seed evaluation. All four means are lower, and every exploratory paired 95% interval includes zero. All four NF4 quantizer-only controls failed during initialization, so same-base comparisons that isolate the initializer are unavailable.

| Recipe | Test JSON exact match %, mean ± sample SD |
|---|---:|
| Accepted DoRA, BF16 base | 73.83 ± 2.95 |
| NF4 i1, original magnitude | Unavailable: initialization failure |
| LoftQ i1, original magnitude | 69.01 ± 2.15 |
| NF4 i1, quantized magnitude | Unavailable: initialization failure |
| LoftQ i1, quantized magnitude | 68.75 ± 1.70 |
| NF4 i3, original magnitude | Unavailable: initialization failure |
| LoftQ i3, original magnitude | 72.14 ± 0.81 |
| NF4 i3, quantized magnitude | Unavailable: initialization failure |
| LoftQ i3, quantized magnitude | 67.97 ± 4.50 |

![Dense-NF4 Qwen/ViGGO comparison, fixed and tuned lanes](quantized_extraction/quantized_extraction.png)

These rank-8 Qwen2.5-3B-Instruct runs use deterministic subsets of the official ViGGO splits: 1,024 training / 128 validation / 256 test examples after seven training duplicates were excluded. This is a subset evaluation. Settings are selected by validation NLL; every successful family selected LR `0.0003`, so fixed and tuned lanes share checkpoints. Each LoftQ family searches four initializer settings × two rates; the anchor and each quantizer control search two rates. Values use seeds 42–44.

LoftQ uses one or three iterations (i1/i3), with magnitudes initialized from the original or quantized weights. The comparison against accepted DoRA combines quantization and initialization. This dense dequantized NF4 simulation provides no compressed-memory or serving-speed measurement. The closest mean is LoftQ i3 with original magnitudes: −1.69 points versus the BF16 anchor, paired 95% interval [−8.07, +4.82]. Intervals use 10,000 paired seed/example resamples without multiple-comparison correction.

All 42 pilot attempt receipts preceded final test evaluation: 34 successful 48-update fits and eight initialization failures. Final receipts cover 15 successful fresh 256-update fits and 12 failed paired-control attempts. The 20 zero-update failure receipts propagate failed initial probes; they are not independently executed training loops and are never assigned zero accuracy. No pilot training is reused.

The [full Qwen quantization report](quantized_extraction/README.md) retains all nine families in both lanes, selected settings, secondary metrics, failures and costs. Its corrected final audit checked 51 successful fits including controls, 5,482 updates, 11,016 checkpoint tensors and 3,840 test predictions. Independent raw scoring verifies the displayed means and sample SD; portable replay reproduces all 15 report outputs byte-for-byte. Test NLL remains the original inference measurement. The public implementation is unchanged; remaining main-grid tasks, quantized Aircraft and main-only fresh-seed confirmation are pending.

