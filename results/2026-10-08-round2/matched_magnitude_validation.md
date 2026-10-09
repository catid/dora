This is a validation-only optimizer diagnostic at matched factor learning rate, rank, seed and trial budget. It does not measure test superiority or change any selection.

| Task | Validation metric | Magnitude LR multiplier | Matched validation pairs | Strictly better / worse / tied | Median benefit in validation units |
|:--|:--|--:|--:|:--|--:|
| aircraft | validation_macro_accuracy | 0.01 | 4 | 2 / 2 / 0 | -0.00224596 |
| aircraft | validation_macro_accuracy | 0.1 | 4 | 2 / 2 / 0 | -0.000971466 |
| cogs | validation_target_token_nll | 0.01 | 2 | 0 / 2 / 0 | -0.00389158 |
| cogs | validation_target_token_nll | 0.1 | 2 | 0 / 2 / 0 | -0.00232573 |
| retrieval | validation_ndcg_at_10 | 0.01 | 4 | 1 / 3 / 0 | -0.000215618 |
| retrieval | validation_ndcg_at_10 | 0.1 | 4 | 1 / 3 / 0 | -0.000309337 |
| teacher | relative_validation_mse | 0.01 | 56 | 6 / 50 / 0 | -0.0320903 |
| teacher | relative_validation_mse | 0.1 | 56 | 14 / 42 / 0 | -0.00242871 |

Positive benefit means lower validation MSE/NLL or higher validation accuracy/nDCG, as appropriate. Metrics have different units and are not combined across tasks.

Exploratory optimizer diagnostic at one tuning seed. Shared-LR cases and synthetic cells are not independent statistical replicates; no confidence interval or test-superiority inference is made. Strict better/worse counts include arbitrarily small floating-point differences. Candidate methods may choose different validation-best checkpoints. Joint-search conclusions remain conditional on tested grids and horizons.
