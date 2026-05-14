# Reviewer comments

## Reviewer 1

### Remarks on code availability

- I downloaded the GitHub repo, and looked at the instructions. It required conda, and I don't want conda on my computer (after strong negative experiences).

### Reviewer expertise

- spatio-temporal modelling, statistics

The paper addresses the problem of modelling alpha and beta diversity by analysing the PREDICTS dataset, which summarizes 25,987 species inventories from 681 studies. It distinguishes between “interpolation,” where cross-validation ignores the distinction between studies and prediction takes place based on model training using the same studies, and “extrapolation,” where complete studies are either placed in training or testing, meaning prediction takes place from observations from other studies.

This paper measures accuracy using Pearson’s correlation. This can be done, but it misses the point that one can have a perfect correlation (1) between predicted and observed values and still have a bias (MAE not equal to zero), or a conditional bias (the points lie on a straight line, but this line does not correspond to x = y). I missed some discussion of that. Also, a more commonly used measure is the square of r, R², which can be interpreted as the fraction of the variance explained by the model; it is used in Figure 3d. Should this be pointed out? If I were the author, I would use R² everywhere. Related to this is the fact that the authors note that Figures 3c and 5c “show very different patterns” (line 207), although r and MAE are practically identical. This is a clear case that these two measures are not enough to describe what is going on.

I do not think I agree with the interpretation of the conditional bias in Figure 5b (page 8, lines 179–181): this is simply conditional bias, and that is what regression always does (“regression to the mean”).

As a person with a spatial statistics background, I think the strongest problem I have with this paper is that it considers the dataset as the result of an experiment, and analyses it both while ignoring the study effect and while incorporating the study effect, then tries to interpret the results of incorporation (“extrapolation”) as a country effect. I agree that each study is probably carried out in a country, but although individual studies may be carried out using some form of experimental design, the collection of studies does not form a design (see Figure 1 in Hudson et al.). Some countries are small, while others cover several climate zones. Some discussion of how the translation of results from looking within versus between studies leads into results concerning biodiversity assessment for countries would have been appropriate, especially in relation to the lack of design when compiling PREDICTS, but I missed that.

Clarification on last paragraph: I think my question boils down to this: for the analysis path chosen, I believe that ideally, the data would come from some form of a designedand controlled experiment. However, no such experiment was carried out: observational data were collected, and an experimental structure (study, country) was put onto it. Question one is: how could that have affected the results?

Question two looks into "country" as a factor: well, there are small
countries, large countries, rich ones and poor ones (as of having
funding for and investing in biodiversity research), and countries with
or without strong biophysical gradients (differences) inside their
territories. Does any of that affect the analysis results?

I don't reject the analysis path chosen, I'm just interested in whether
the authors also see these challenges, and if so what their thoughts are
about them. Maybe they clearly see these as non-issues and can point out
why they consider them as non-issues? If they can't relate to them, then
please ignore

### Minor issues

- In Figure 2, the color of the “Training” class in the legend does not correspond to the color used in the figures.
- Page 6, line 170 mentions Figure 4c. Should this be 4b? Figure 4c does not exist.
- In Figures 3 and 5, the caption does not explain what the smooth red curve reflects.
- In the Figure 5 caption, “hlfive nearest” must be a typo.
- The Figure 6 caption (or legend) does not explain which accuracy measure is shown. Given that most of the paper discusses Pearson’s r, it may be that, but it is worth mentioning explicitly in the caption.
- Page 13, line 339: typo “notn”.
- Figure 6 uses a Plate Carrée projection (and ignores Antarctica), which distorts areas. An equal-area projection should be used.

## Reviewer 2

I enjoyed reading this manuscript by Jakob Nyström and colleagues. The authors aimed to compare the interpolation and extrapolation accuracy of two correlative global models - trained on the same dataset – for predicting the impacts of human pressures (e.g., land use, road proximity) on local biodiversity, using two biodiversity metrics (geometric mean abundance and Bray-Curtis similarity index). They found generally low to medium interpolation accuracy and low extrapolation accuracy. I considered the topic and research question quite relevant in view of the key role of global biodiversity modelling in support of policy. Further, I found the paper easy to follow and the figures well-designed and illustrative. Nevertheless, I also recognize some shortfalls and weaker points, which I think require revision or reconsideration, as explained below.

### General or main comments

The framing of the paper in the context of monitoring appears a bit misleading to me. I typically understand monitoring as the collection of new primary field data (whether through direct measurement on site or through remote sensing), while the current study is about predictive modelling. Although I acknowledge that the results of pressure-impact models have the potential to be used to track progress towards policy goals, I would not call this monitoring. Perhaps frame differently or make explicit how you understand monitoring?

The design of the study appears somewhat problematic in the sense that three aspects are being changed from model 1 to model 2 (frequentist -> Bayesian approach, different random effects structure, and additional predictors). This hampers the understanding of where performance differences arise from, as the authors also acknowledge themselves (e.g., in line 182-183, ‘It should be noted that the structural differences between the models imply some limitations on direct intercomparison’). Perhaps it does not matter an awful lot in the end, as performance differences between the two models are relatively small anyway. Nevertheless, I would find it more intuitive and cleaner if a more systematic, factorial design was chosen for the comparison. Would this be feasible and worth considering?

Based on my understanding, no model/predictor selection was performed. I think this is problematic in the sense that I suspect some collinearity in the predictor set, in particular among the same predictor defined for different buffer sizes (e.g., road or population density at 1 km and at 50 km). I would suggest either a systematic model selection procedure (if feasible given the size and complexity of the models) or else at least test for collinearity among the predictors and remove those above a collinearity threshold (e.g., VIF > 3).

I’m not convinced by the choice of Pearson’s r as a model performance measure. Because Pearson’s r is designed to measure linear association, it will yield a high value as long as the ranking/order of predictions and observations are aligned. However, it ignores possible systematic bias. For example, consider a case where your predictions would be 0.5, 0.6, 0.7 and 0.8 while your observations were 0.3, 0.4, 0.5 and 0.6. Pearson’s r would tell you the model is perfect, while I would argue that your model is systematically off. I think it would be better to use a measure of accuracy (rather than association), such as the Nash–Sutcliffe efficiency (in addition to the MAE already included, which is fine).

I wonder why you used the gROADS dataset to estimate road density, rather than a newer (and more complete) alternative (e.g., OpenStreetMap or GRIP). I acknowledge that OSM is very challenging to handle, but GRIP provides a global road density raster (5 arc-min resolution), so that should be quite straightforward to use.

I struggled a bit with the interpretation and implications of the findings. Indeed, the predictive accuracy of the models is quite low, but I think this is quite common for (global) meta-analytical biodiversity models, which tend to have a much lower marginal R2 than conditional R2. This implies that we cannot use the regression coefficients or effect sizes of such models to predict a biodiversity metric value at a specific site measured in a specific study, but then again, should this actually be the goal? Isn’t the purpose of the type of model presented here to predict broad-scale patterns and trends in biodiversity rather than site- and study-specific values? I miss a reflection on this tension in the discussion, as well as concrete recommendations for improvements. To what extent should residual heterogeneity be reduced and how? Which biases need to be tackled to ensure that the overall mean regression coefficients are meaningful? The current paragraph of recommendations (from line 281) appears a bit superficial and in places a bit off-topic. Notably, it is unclear to me how data from GBIF (often opportunistic, spatially biased and at the species level) could resolve the issue of low predictive power of community-level models as observed here. More concrete recommendations would be helpful.

### Specific/minor comments

- Title: in line with my comment above, the word ‘monitoring’ seems a bit misleading, as your study describes a modelling effort. Could you replace ‘monitoring’ with ‘modelling’?
- Line 30 - ‘the critical role of biodiversity indicators to halt and reverse this development’: indicators per se are not a means to halt or reverse loss; what matters is the conservation actions we take (where indicators can help us track effects of these actions). Please rephrase.
- Line 34-35 – ‘lingering geographic and taxonomic gaps make comprehensive biodiversity monitoring challenging’: this appears circular to me, as the geographic and taxonomic gaps are monitoring gaps, no? So, the sentence seems to say that gaps in monitoring data challenge monitoring. Would be good to rephrase.
- Line 59-60 – ‘there has also been critique against lack of agreement with other global metrics’: which metrics? And why is that an issue given that different metrics might be purposedly designed to capture different aspects of biodiversity (hence the lack of agreement is by design)? More explanation would be helpful here.
- Line 71 – ‘nonetheless important’: maybe even more important? ‘nonetheless’ seems a bit misplaced.
- Line 91: I don’t think GMA is indicative of richness, as the abundance values are normalized by the number of species. I would argue that GMA represents abundance and evenness (see also e.g. Buckland et al. 2011). Similar comment for line 344.
- Line 103 – ‘Spatial block intercepts further captured intra-study variation where available’: why these spatial blocks (a reference for this would be helpful) and how were they delineated?
- Line 282-284: it is unclear to me how GBIF data – often opportunistic, spatially biased and at the species level – may help resolve the limited predictive power of global community-level biodiversity models (see also my general comment above).
- Line 284 – Facilitaty -> Facility
- Line 286 – 287: What methodological developments exactly? More concrete recommendations would be helpful.
- Line 294: the notion of ‘good enough’ is interesting and relevant. While I acknowledge that establishing a final definition of ‘good enough’ might be beyond the goal and scope of the current study, I think a reflection on possible criteria would be very informative (either in this paragraph or earlier on in the discussion). Could we say, for example, that a model is good enough if i) it is based on a representative, unbiased sample of observations and ii) its predictions are better than a grand average across the observations (as expressed by a Nash–Sutcliffe efficiency value > 0)?
- Line 343: beta diversity metrics -> beta diversity metric
- Line 350-351 - ‘For a given total site abundance, higher species richness and evenness results in a greater GMA.’ This should be lower species richness I think (as S is in the denominator).
- Line 453-454 – ‘before generating circular polygons of different spatial extents’: could you provide more detail here?
- Line 457 – ‘after which the polygons were reprojected to the global format’: why is that needed?
- Line 509-511: which accuracy measure?
- Line 522-523: what are the taxonomic groups distinguished?
- Line 611-618: see general comment above about Pearson’s r.
- Fig. 3b and 3c, 5b and 5c: just noting that the axes lack labels.
