# HPO with Optuna

FujiCV integrates with [Optuna](https://optuna.org) for hyperparameter optimisation.

```bash
pip install "fujicv[hpo]"
```

## Basic search

```python
from fujicv.hpo import run_hpo

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 0, 1e-2)

    model = ModelBuilder("resnet18", task="classification", num_outputs=10).build()
    trainer = Trainer(
        model=model, ...,
        optimizer=optim.AdamW(model.parameters(), lr=lr, weight_decay=wd),
        epochs=5,
    )
    history = trainer.train()
    return max(history.metrics.get("val_accuracy", [0]))

result = run_hpo(objective, n_trials=30, direction="maximize")
print(result["best_params"])
```

## With pruning

Cut unpromising trials early to save compute:

```python
result = run_hpo(
    objective,
    n_trials=50,
    pruner="median",             # or "hyperband", "percentile", "successive_halving"
    pruner_kwargs={"n_startup_trials": 5, "n_warmup_steps": 2},
)
```

## Visualize results

```python
from fujicv.hpo import plot_optimization_history, plot_param_importances

fig1 = plot_optimization_history(result["study"])
fig1.savefig("history.png")

fig2 = plot_param_importances(result["study"])
fig2.savefig("importances.png")
```

## Persist across restarts

```python
result = run_hpo(
    objective,
    n_trials=100,
    storage="sqlite:///hpo.db",   # resume from DB on restart
    study_name="my_experiment",
)
```
