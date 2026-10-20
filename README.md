# P1 - Steel Production Data

Applied Machine & Deep Learning (190.015), WS 2026/27. Predict the steel quality factor `output` from 21 process measurements (`input1`..`input21`) - a regression task.

## Data and provenance
Downloaded from the course share <https://cloud.cps.unileoben.ac.at/index.php/s/xtsCHycaSiHsqHT> (`pr1_steel_quality_prediction_with_DNNs`): `normalized_train_data.csv` (7642 x 22) and `normalized_test_data.csv` (3337 x 22). Files are **not committed** (`data/raw/*.csv` is git-ignored; licence unclear) - download and place them in `data/raw/`. Template (structure only) from <https://cloud.cps.unileoben.ac.at/index.php/s/KDGztAaH6LbHf4e>.

Findings from inspection: all columns numeric, **no missing values, no duplicates, no categorical columns**; every column in both files already has min 0 / max 1, i.e. the course normalised them, apparently **each file separately** (train and test column means differ strongly, e.g. `input3` 0.90 vs 0.36). Physical units/names are unknown.

## Run
```
pip install -r requirements.txt
python -m pytest
python src/scripts/run_pipeline.py
```
Outputs: `results/tables/` (data_report.json, correlations.csv, model_comparison.csv) and `results/figures/` (histograms, correlation with output, model comparison bar plot with mean ± std).

## Pipeline (slide steps)
1. Duplicate removal, missing-value check (counts reported; imputation code in place but unused by this data).
2. Histograms and correlation of inputs with `output` (train data).
3. Train 80% / validation 20% split of the train file (seed 42); provided test file kept as test set.
4. Outliers: clipped at 3xIQR fences fitted on the train split.
5. Categorical encoding (one-hot) and scaling (StandardScaler) are fitted on train only, inside sklearn Pipelines (no leakage). No categorical columns exist here.
6. Models: Ridge, Random Forest, SVR. 5-fold CV mean ± std (RMSE, MAE, R2) on the train split, plus validation and test scores.

## Results (from the run, seed 42; `results/tables/model_comparison.csv`)
| Model | CV RMSE | CV R2 | Val R2 | Test R2 |
|---|---|---|---|---|
| Ridge | 0.0751 ± 0.0015 | 0.196 ± 0.016 | 0.110 | -1.895 |
| Random Forest | 0.0610 ± 0.0019 | 0.469 ± 0.017 | 0.428 | -2.297 |
| SVR | 0.0705 ± 0.0020 | 0.291 ± 0.034 | 0.224 | -1.206 |

Random Forest is best on CV and validation. **All models fail on the provided test file (negative R2)**. Likely cause (hypothesis, not verified): train and test were min-max scaled separately / come from a different distribution, so the feature meanings differ. Ask the course whether the test file is meant to be on the train scale; do not trust test numbers until clarified.

## Limitations / status
- Done: pipeline, tests (5 passing), tables and figures.
- Not done (needs you): short report text (related work, motivation), 6-10 slides, LICENSE choice, ChatGPT-prompt acknowledgements, optional MLP/DNN comparison and hyperparameter tuning.
- Models use default-ish hyperparameters; no tuning.
