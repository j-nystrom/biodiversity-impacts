# Ecological Structure Alpha Training Experiments

Run these experiments from the matching branch with:

```bash
python experiments/run_experiment.py \
  --experiments-file experiments/ecological_structure/<file>.yaml
```

The experiment name is used as the run-folder suffix. Runtime is written to
`experiment.log`; performance metrics are saved under `key_output/`, for
example `train_metrics.pkl`.

Each YAML file represents one model structure and contains two training runs:
one prediction run with fitted study/block effects included where supported,
and one fit-only proxy run where those effects are omitted from predictions.

Control-branch caveat: in `revisions-control`, `rolled_up_predictions=True`
both removes study/block effects from prediction and applies ecological roll-up
fallback. The two control runs therefore are not a pure study-effect contrast.
