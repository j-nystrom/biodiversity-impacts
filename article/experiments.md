# Experiments

## Testing

### Rolled-up Bayesian ecological hierarchy

Purpose: test the new training-time roll-up of ecological groups against the
previous full-hierarchy implementation.

Model structure and effects used:

- Response/model: alpha diversity, Bayesian hierarchical model, beta likelihood.
- Ecological hierarchy: `level_1 = Biome + Custom_taxonomic_group`;
  `level_2 = Biome + Custom_taxonomic_group + Realm`.
- Fitted ecological levels: 2.
- Varying ecological slopes fitted to level 2.
- Training-time roll-up: `train_on_rolled_up_groups = True`.
- Minimum study threshold: `min_studies_per_group = 5`.
- Training components fitted: ecological effects, study intercepts, study
  slopes, and SSB block intercepts.
- Study slope terms: the configured study-level slope controls for land-use and
  land-use intensity pressure variables, plus `Pop_density_10km_log` and
  `Road_density_10km_log`.
- Prediction components applied in this experiment: ecological effects and
  study slopes. This is therefore not the ecological-only deployable prediction
  setting.
- Sampler: 4 chains, 100 tuning + 100 posterior draws per chain.

Performance:

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.525 |
| R2 (variance explained) | 0.573 |
| Mean absolute error | 0.132 |
| Median absolute error | 0.102 |
| Pearson correlation | 0.730 |
| Spearman rank correlation | 0.745 |
| Bias ratio (pred/obs) | 1.041 |

Runtime:

| Chain | Runtime | Seconds per iteration |
| --- | ---: | ---: |
| 0 | 24:37 | 7.39 |
| 1 | 24:54 | 7.47 |
| 2 | 25:01 | 7.51 |
| 3 | 24:16 | 7.28 |

### Original full-hierarchy Bayesian model

Purpose: benchmark the previous implementation against the new training-time
roll-up model.

Model structure and effects used:

- Response/model: alpha diversity, Bayesian hierarchical model, beta likelihood.
- Ecological hierarchy: same biome-taxon and biome-taxon-realm structure.
- Full ecological hierarchy fitted during training, with roll-up/fallback applied
  only after training for prediction.
- Training controls: study and SSB block intercepts, matching the old
  implementation.
- Study slopes: not fitted in this comparison.
- Prediction components: ecological effects used for deployable predictions;
  study/block controls treated as training controls.
- Sampler: 4 chains, 100 tuning + 100 posterior draws per chain.

Performance:

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.536 |
| R2 (variance explained) | 0.575 |
| Mean absolute error | 0.130 |
| Median absolute error | 0.102 |
| Pearson correlation | 0.737 |
| Spearman rank correlation | 0.755 |
| Bias ratio (pred/obs) | 1.047 |

Runtime:

| Chain | Runtime | Seconds per iteration |
| --- | ---: | ---: |
| 0 | 18:13 | 5.47 |
| 1 | 18:01 | 5.41 |
| 2 | 17:55 | 5.38 |
| 3 | 18:08 | 5.44 |

### Diagnostic: study and block intercepts in prediction

Purpose: test in-sample prediction when study and SSB block intercepts are used
both for training and prediction.

Model structure and effects used:

- Response/model: alpha diversity, Bayesian hierarchical model, beta likelihood.
- Ecological hierarchy: rolled-up biome-taxon and biome-taxon-realm structure.
- Training components fitted: ecological effects, study intercepts, and SSB block
  intercepts.
- Study slopes: not fitted in this run.
- Prediction components applied: ecological effects, study intercepts, and SSB
  block intercepts.
- Sampler: 4 chains, 100 tuning + 100 posterior draws per chain.

Performance:

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.268 |
| R2 (variance explained) | 0.585 |
| Mean absolute error | 0.158 |
| Median absolute error | 0.118 |
| Pearson correlation | 0.641 |
| Spearman rank correlation | 0.664 |
| Bias ratio (pred/obs) | 0.942 |

Runtime:

| Chain | Runtime | Seconds per iteration |
| --- | ---: | ---: |
| 0 | 18:52 | 5.66 |
| 1 | 19:14 | 5.77 |
| 2 | 18:47 | 5.64 |
| 3 | 19:29 | 5.85 |

Interpretation: this result should be treated as a diagnostic failed run, not a
valid benchmark. The prediction graph was applying both `gamma_study` and
`gamma_block`, but `gamma_block` is already nested on `gamma_study` in the
training model. This double-counted the study intercept whenever both
`study_intercept` and `block_intercept` were enabled for prediction.

### Initial interpretation

The rolled-up training model produced very similar in-sample performance to the
original full-hierarchy model, with slightly lower Pearson correlation, slightly
lower standard R2, and slightly higher MAE. The difference is small enough that
it should not be interpreted strongly from one short test.

Runtime was worse for the rolled-up model in this test, despite fitting fewer
ecological group parameters. That should be checked before presenting the
training-time roll-up as a computational improvement. Possible explanations
include sampler geometry, changed model structure, study/block controls, and
ordinary run-to-run variation at 100 + 100 iterations.
