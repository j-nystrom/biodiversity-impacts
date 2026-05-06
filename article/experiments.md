# Experiments

## Structured Ecological Model Runs

These runs compare alpha-diversity Bayesian hierarchical model structures under
the structured experiment setup. Unless noted otherwise, the shared base setup is
alpha diversity, biome-taxon plus realm ecological hierarchy, roll-up threshold
of 5 studies, and 500 tuning + 500 posterior iterations.

### 1. Old model with study-block intercepts

Branch/model: old Bayesian model structure with study and block intercepts.

#### Rolled-up predictions

Prediction setting: rolled-up predictions enabled.

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.164 |
| R2 (variance explained) | 0.339 |
| Mean absolute error | 0.185 |
| Median absolute error | 0.157 |
| Pearson correlation | 0.469 |
| Spearman rank correlation | 0.486 |
| Bias ratio (pred/obs) | 1.112 |

Runtime: 2:44:30.

#### Without rolled-up predictions

Prediction setting: rolled-up predictions disabled.

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.536 |
| R2 (variance explained) | 0.575 |
| Mean absolute error | 0.130 |
| Median absolute error | 0.102 |
| Pearson correlation | 0.737 |
| Spearman rank correlation | 0.755 |
| Bias ratio (pred/obs) | 1.047 |

Runtime: 2:43:49.

#### Initial note

The non-rolled prediction run has substantially better in-sample performance
than the rolled-up prediction run, while runtime is effectively unchanged. This
is expected to some extent because rolled-up predictions remove lower-level
group-specific information for sparse ecological groups.

### 2. New model with study-block intercepts

Branch/model: new Bayesian model structure equivalent to the old model, with
study and block intercepts fitted.

#### Study/block effects used for prediction

Prediction setting: study and block intercepts included in prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.514 |
| R2 (variance explained) | 0.566 |
| Mean absolute error | 0.133 |
| Median absolute error | 0.104 |
| Pearson correlation | 0.723 |
| Spearman rank correlation | 0.741 |
| Bias ratio (pred/obs) | 1.045 |

Runtime: 2:14:48.

#### Study/block effects not used for prediction

Prediction setting: study and block intercepts fitted, but excluded from
prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | -0.218 |
| R2 (variance explained) | 0.311 |
| Mean absolute error | 0.227 |
| Median absolute error | 0.200 |
| Pearson correlation | 0.320 |
| Spearman rank correlation | 0.320 |
| Bias ratio (pred/obs) | 1.312 |

Runtime: 2:14:41.

#### Initial note

Using study/block effects for prediction gives performance close to the old
non-rolled prediction run. Excluding those effects causes a large drop in
standard R2 and increases error, which is consistent with this setting acting as
a stricter proxy for out-of-study prediction performance.

### 3. New model with pressure study slopes

Branch/model: new Bayesian model structure with study and block intercepts plus
study-level slopes for pressure variables.

#### Pressure slopes used for prediction

Prediction setting: study and block intercepts plus pressure slopes included in
prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.560 |
| R2 (variance explained) | 0.597 |
| Mean absolute error | 0.126 |
| Median absolute error | 0.098 |
| Pearson correlation | 0.752 |
| Spearman rank correlation | 0.767 |
| Bias ratio (pred/obs) | 1.040 |

Runtime: 2:16:04.

#### Pressure slopes not used for prediction

Prediction setting: study and block intercepts plus pressure slopes fitted, but
excluded from prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | -0.189 |
| R2 (variance explained) | 0.313 |
| Mean absolute error | 0.223 |
| Median absolute error | 0.195 |
| Pearson correlation | 0.317 |
| Spearman rank correlation | 0.322 |
| Bias ratio (pred/obs) | 1.279 |

Runtime: 2:16:06.

#### Initial note

Including pressure slopes in prediction improves in-sample performance over the
intercept-only new model. When the pressure slopes are fitted but excluded from
prediction, performance remains close to the intercept-only fit-only proxy,
suggesting the additional study-level pressure slopes do not materially improve
this stricter prediction setting.

### 4. New model with full study random effects

Branch/model: new Bayesian model structure with study and block intercepts plus
study-level slopes for pressure and environmental covariates.

#### Full random effects used for prediction

Prediction setting: study and block intercepts plus all study-level slopes
included in prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | 0.567 |
| R2 (variance explained) | 0.603 |
| Mean absolute error | 0.125 |
| Median absolute error | 0.097 |
| Pearson correlation | 0.757 |
| Spearman rank correlation | 0.771 |
| Bias ratio (pred/obs) | 1.039 |

Runtime: 2:08:38.

#### Full random effects not used for prediction

Prediction setting: study and block intercepts plus all study-level slopes
fitted, but excluded from prediction.

| Metric | Value |
| --- | ---: |
| R2 (standard) | -0.182 |
| R2 (variance explained) | 0.314 |
| Mean absolute error | 0.224 |
| Median absolute error | 0.196 |
| Pearson correlation | 0.315 |
| Spearman rank correlation | 0.321 |
| Bias ratio (pred/obs) | 1.269 |

Runtime: 2:08:22.

#### Initial note

The full random-effects model gives the strongest in-sample performance among
the new-code runs when all fitted effects are used for prediction. Excluding
those effects again leaves performance close to the other fit-only proxy runs,
with little improvement from adding environmental random slopes.
