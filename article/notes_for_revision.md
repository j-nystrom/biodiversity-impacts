# Notes for revision

## Results and figure structure

The revised Results should stay close to reviewer-motivated changes rather than
introducing a substantially new manuscript structure. The main storyline should
focus on the structure and limitations of PREDICTS, the difference between
deployable prediction components and training-only controls, and the contrast
between interpolation and extrapolation performance.

### Fig 1: PREDICTS data structure

Focus on the structure of the PREDICTS database and the implications for model
design. The figure should show that the dataset is broad but highly uneven:
study sizes vary strongly, observations are clustered within studies, and
coverage across biome, realm, taxonomic group, and their combinations is
unbalanced. Some existing panels can likely be simplified if they do not support
this argument directly.

### Fig 2: Overall predictive performance

Keep the same basic figure structure as the current version, with one panel for
each response: alpha diversity, beta diversity, and the two alpha-delta
responses. Within each panel, show the three key metrics across cross-validation
runs:

- Pearson correlation.
- Residual-based R2.
- MAE expressed as a skill metric.

The main figure should probably show only the main model structures. The full
model ladder, additional metrics, and calibration plots can go in Extended Data.

### Fig 3: Interpolation deep dive

The aim is to show why ecological structure improves interpolation. Replace the
current variance-explained framing based on fixed versus random effects in the
SBM with a partition into effects usable at prediction versus effects that are
training-only controls.

For alpha and beta diversity, show variance explained for each of the three main
model structures, split into:

- Prediction-usable components.
- Training-only study/control components.
- Residual variance.

Then show the spread in alpha-diversity effect sizes for the three models. The
SBM has one average estimate, whereas the ecologically structured models can
represent broader ecological heterogeneity. The key message is that ecological
structure captures more of the heterogeneity that otherwise appears as study
variation, which can improve interpolation.

Calibration plots should move to Extended Data.

Open issue: the most correct variance partition may require fitting the full
study-effect structure also for the ecological models. However, study effects are
regularized through priors, so the amount of study heterogeneity is not a fixed
reference across model structures. One alternative is to use the SBM study-effect
structure as a reference, then fit the ecological models with the older
intercept-only study structure and overlay their ecological parameters. This
mechanically works for effect-size comparisons. A related approach might also be
possible for variance explained, but it would need to be explained explicitly as
a reference decomposition rather than a fully model-internal variance partition.

Current preference: use a model-internal variance partition in the main figure,
provided the model structures are stated clearly up front. For each model, split
variance into prediction-usable ecological components, training-only study/block
components, and residual variance. This means the study-control component is not
a fixed reference across models, but that is also substantively meaningful:
ecological structure may absorb heterogeneity that otherwise appears as study
variation. An SBM-referenced decomposition can be used as a sensitivity analysis
or interpretive overlay, but should not be presented as the primary
model-internal partition.

For the variance-explained analysis, it may be preferable to fit study slopes for
all model covariates, not only pressure variables, if the goal is to estimate
study-specific heterogeneity as completely as possible. Otherwise, only pressure
effects get a study-specific control term, while environmental covariate
heterogeneity is forced into ecological effects, population effects, or the
residual. This broader study-slope model should probably be treated as a
variance-decomposition sensitivity rather than the default prediction model,
because it adds many weakly identified training-only terms.

Prior tightness matters for this interpretation. Study effects are controls, not
deployable prediction terms, so their priors should be regularizing rather than
loose. Study intercepts can be moderately regularized because baseline shifts are
usually identifiable. Study slopes should be more strongly regularized and
centered at zero, especially if slopes are fitted for all covariates. Ecological
slopes can be less tightly regularized than study slopes because they are part of
the scientific and prediction target. Include at least one tighter and one looser
study-slope prior sensitivity in Extended Data if these controls become central
to the revised variance-explained figure.

### Fig 4: Extrapolation breakdown

The aim is to show why model flexibility can become fragile under extrapolation.
First show distribution shifts in covariates for alpha and beta diversity. Then
show how estimated effects vary across folds for the three main model
structures.

For the SBM, this is straightforward: show the estimated fixed effects per fold.
For the ecologically structured models, the right summary is less obvious.
Showing only population-level effects may hide the fold discrepancies that arise
from flexible group-level responses and that may be responsible for weaker
extrapolation performance.

Current preference: compare fold-to-fold deviations in the deployable
group-specific pressure effects for the same retained ecological groups. For a
group, pressure, and fold, define the deployable effect as the population slope
plus the ecological group deviation actually used after roll-up. Then summarize
the absolute deviation between folds for groups that are retained in at least two
folds. This is easier to explain than prediction-scenario contrasts and avoids
the misleading population-only summary.

The coefficients should be placed on a comparable scale before plotting. If
continuous predictors are standardized separately within folds, raw coefficients
partly reflect different fold-specific scaling. Prefer either back-transformed
effects on the original covariate scale or a standardized pressure contrast,
such as the expected change from the 10th to 90th percentile. The main panel
could show mean absolute fold-to-fold deviation by model and pressure variable,
with the spread across ecological groups shown as intervals or points.

Possible alternatives:

- Show fold-to-fold variation in predictions under standardized pressure
  scenarios for representative biome-realm or biome-taxon groups.
- Show distributions of group-level slope deviations across folds, with partial
  pooling preserved.
- Show performance or prediction instability as a function of ecological group
  novelty or covariate shift.
- Use population-level effects only as a conservative summary, but state clearly
  that this under-represents group-level flexibility.

The expected message is not simply that the SBM is better. Rather, the simpler
SBM may be more stable under strong distribution shift because it has fewer
group-specific degrees of freedom. Ecological flexibility should help most when
the ecological context of the test data is represented in training.

Calibration plots should move to Extended Data.

### Fig 5: Biome-realm performance map

Revise the current map framing so that performance is shown by biome-realm
rather than by country. This aligns the map with the ecological model structure
and avoids implying country-level inference.

Potential layout:

- Alpha diversity, standard CV.
- Alpha diversity, cross-study CV.
- Beta diversity, standard CV.
- Beta diversity, cross-study CV.

If four maps are too crowded, prioritize cross-study CV in the main figure and
move standard CV maps to Extended Data.

### Fig 6: Drivers of performance differences

Revise the current Fig 5 to focus on potential drivers of performance
heterogeneity. Candidate drivers include:

- Number of studies or observations per biome-realm.
- Environmental or pressure novelty.
- Land-use/covariate coverage.
- Imbalance across biome-realm-taxon groups.
- Dominance by a small number of large studies.

This should be framed diagnostically rather than causally. The figure can show
associations between data structure, novelty, and performance, but should avoid
overclaiming that these factors fully explain performance differences.

## Extended Data candidates

- Full model ladder, including secondary model structures.
- Full metric table for Pearson r, residual R2, MAE skill, and bias.
- Overall, top, and bottom bias diagnostics.
- Calibration plots for main models.
- Sensitivity to study slopes, block intercepts, and rolled-up ecological groups.
- Convergence and runtime summaries.
- Additional data imbalance plots if Fig 1 becomes too crowded.
