# From Zero to MLflow: Tracking, Tuning, and Deploying a Keras Model (Hands‑on)

I practiced MLflow end to end across two notebooks folders (`1-MLproject` and `2-DLproject`) to learn how to track experiments, compare runs, pick the best model, and serve it. This article distills everything I learned, with code snippets you can copy and run.


## What you’ll build

- Run and track training experiments with MLflow
- Log parameters, metrics, and models from Keras/TensorFlow
- Tune hyperparameters with Hyperopt and compare runs in the MLflow UI
- Pick the best run, register a model, and serve it via a local REST API

Repo layout used:
- `1-MLproject/` — classic ML experimentation (e.g., house price notebook)
- `2-DLproject/` — deep learning quickstart with MLflow tracking + Hyperopt
- `requirements.txt` — dependencies to reproduce my runs


## Environment and data

Install deps and launch the UI (optional but helpful while you work):

```bash
pip install -r requirements.txt
mlflow ui --port 5000
```

Dataset used: White wine quality from the MLflow repo.

```python
import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)
```

Train/valid/test split:

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.25, random_state=42)

train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()

test_x = test.drop(["quality"], axis=1).values
test_y = test["quality"].values.ravel()

train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.20, random_state=42
)
```

We’ll also infer a model signature for safer serving later:

```python
import mlflow
from mlflow.models import infer_signature

signature = infer_signature(train_x, train_y)
```


## Set up MLflow experiment

```python
import mlflow

mlflow.set_experiment("/wine-quality")  # Creates/uses this experiment namespace
```

This groups your runs in the UI under the “wine-quality” experiment.


## A simple Keras model with MLflow logging

Here’s the exact training function I used in `2-DLproject/starter.ipynb`. It trains a small network and logs params, metrics, and the model to MLflow. Each hyperparameter trial is a nested MLflow run.

```python
import numpy as np
import keras
from hyperopt import STATUS_OK


def train_model(params, epochs, train_x, train_y, valid_x, valid_y):
    # Compute normalization stats once
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)

    model = keras.Sequential([
        keras.Input([train_x.shape[1]]),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=params["lr"], momentum=params["momentum"]
        ),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # One nested run per trial
    with mlflow.start_run(nested=True):
        model.fit(
            train_x, train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=64,
            verbose=0,
        )

        eval_loss, eval_rmse = model.evaluate(valid_x, valid_y, batch_size=64, verbose=0)

        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", float(eval_rmse))

        # Log the TensorFlow model
        mlflow.tensorflow.log_model(model, "model", signature=signature)

        return {"loss": float(eval_rmse), "status": STATUS_OK, "model": model}
```


## Hyperparameter tuning with Hyperopt (and MLflow tracking)

Define the search space and objective that delegates to `train_model`:

```python
from hyperopt import fmin, tpe, hp, Trials

space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    # Momentum in [0, 1] — use uniform, not loguniform
    "momentum": hp.uniform("momentum", 0.0, 1.0),
}


def objective(params):
    return train_model(
        params=params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
    )
```

Run the sweep under a parent run; then pick and log the best result to the parent run so it’s easy to register later.

```python
mlflow.set_experiment("/wine-quality")

with mlflow.start_run() as parent_run:
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=4,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    # Pick the best trial by validation RMSE
    best_trial = sorted(trials.results, key=lambda x: x["loss"])[0]

    # Log the best params/metrics/model to the parent run
    mlflow.log_params(best_params)
    mlflow.log_metric("eval_rmse", best_trial["loss"])
    mlflow.tensorflow.log_model(best_trial["model"], "model", signature=signature)

    print("Best parameters:", best_params)
    print("Best eval RMSE:", best_trial["loss"])
    parent_run_id = parent_run.info.run_id
```

Open the MLflow UI at http://127.0.0.1:5000, select the “wine-quality” experiment, and compare runs and metrics. The nested runs (one per Hyperopt trial) are great for drill‑down.


## Register the winner in the Model Registry

You can register directly from the UI (select a run → model artifact → “Register Model”), or do it in code using the parent run’s model artifact:

```python
from mlflow import register_model

result = register_model(
    model_uri=f"runs:/{parent_run_id}/model",
    name="finalmodel",
)
print("Registered model:", result.name, "version:", result.version)
```

This will create a registry entry like `models:/finalmodel/1` (you’ll see it in the MLflow UI under “Models”). In the workspace, you’ll notice local registry files under `mlruns/models/finalmodel/`.


## Serve the model as a REST API

Serve a specific version from the registry:

```bash
mlflow models serve -m "models:/finalmodel/1" -p 5001 --no-conda
```

Send a sample request (replace the feature values with your own row; the wine dataset has 11 numeric features):

```bash
curl -s http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8]]}'
```

If you prefer to serve straight from a run (without registry):

```bash
mlflow models serve -m "runs:/$RUN_ID/model" -p 5001 --no-conda
```


## Comparing with a classic ML notebook (house prices)

In `1-MLproject/housepricepredict.ipynb` I followed the same tracking pattern in a non‑DL context: create an experiment, start a run, log inputs (params), outputs (metrics), and persist the model artifact. The MLflow UI experience remains the same: sortable metrics, tags, artifacts, and lineage for reproducibility.

Key steps are identical:

```python
mlflow.set_experiment("/house-prices")
with mlflow.start_run():
    # train your model (e.g., scikit-learn)
    mlflow.log_params({"model": "RandomForest", "n_estimators": 200})
    mlflow.log_metric("rmse", 0.123)
    mlflow.sklearn.log_model(model, "model")
```


## Troubleshooting and gotchas

- Hyperopt distributions: `hp.loguniform` samples `exp(U(low, high))`, so `low`/`high` must be logs of positive numbers. For [0, 1] ranges (like momentum), use `hp.uniform` instead.
- Nested runs: Use `mlflow.start_run(nested=True)` for each trial so the UI keeps the tree under a single parent sweep.
- Signatures: `infer_signature` helps validation at serving time and documents expected input/outputs.
- Reproducibility: Fix random seeds and log data versions and code versions (e.g., `mlflow.set_tag("mlflow.source.git.commit", ...)`).
- UI hiccups: If the UI doesn’t show your latest results, refresh and check you’re in the right experiment.


## What I’d improve next

- Enable `mlflow.autolog()` to capture more metrics automatically
- Add a feature store or data versioning (e.g., Delta Lake/Feast) so data lineage is explicit
- Wire CI to run selected experiments and auto‑register on metric thresholds
- Use Docker for consistent serving environments


## Try it yourself

1) Install and run the sweep

```bash
pip install -r requirements.txt
python - <<'PY'
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import keras

# Data
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)
train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.20, random_state=42)
signature = infer_signature(train_x, train_y)

# Model + objective
import numpy as np
import keras

def train_model(params, epochs=3):
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)
    model = keras.Sequential([
        keras.Input([train_x.shape[1]]),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=params["lr"], momentum=params["momentum"]),
                  loss="mean_squared_error",
                  metrics=[keras.metrics.RootMeanSquaredError()])
    with mlflow.start_run(nested=True):
        model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=64, verbose=0)
        _, eval_rmse = model.evaluate(valid_x, valid_y, batch_size=64, verbose=0)
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", float(eval_rmse))
        mlflow.tensorflow.log_model(model, "model", signature=signature)
        return {"loss": float(eval_rmse), "status": STATUS_OK, "model": model}

space = {"lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)), "momentum": hp.uniform("momentum", 0.0, 1.0)}

def objective(params):
    return train_model(params)

mlflow.set_experiment("/wine-quality")
with mlflow.start_run() as parent_run:
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=4, trials=trials)
    best_trial = sorted(trials.results, key=lambda x: x["loss"])[0]
    mlflow.log_params(best_params)
    mlflow.log_metric("eval_rmse", best_trial["loss"])
    mlflow.tensorflow.log_model(best_trial["model"], "model", signature=signature)
    print("Best:", best_params, "RMSE:", best_trial["loss"])
    print("Parent run:", parent_run.info.run_id)
PY
```

2) Open the UI

```bash
mlflow ui --port 5000
```

3) Serve your registered model

```bash
mlflow models serve -m "models:/finalmodel/1" -p 5001 --no-conda
```

That’s it — you now have a repeatable workflow for tracking, selecting, and serving models with MLflow.
