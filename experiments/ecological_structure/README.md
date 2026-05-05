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
one prediction run with fitted study/block effects included, and one fit-only
proxy run where those effects are omitted from predictions.

This branch contains the new-code model structures. The old-code control file
lives on `revisions-control`.
