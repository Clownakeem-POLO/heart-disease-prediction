# Heart disease prediction on the UCI Cleveland dataset

Predicting whether a patient has heart disease from 13 routine clinical measurements. I
treat this as a binary classification problem, but the framing I care about is clinical:
a false negative is a sick patient sent home, which is worse than a false alarm. So the
metric I optimize for is recall on the disease class, not raw accuracy.

The data is the Cleveland subset of the UCI Heart Disease dataset, 303 patients. Six rows
have missing values in `ca` or `thal`, which I drop, leaving 297. The original target runs
0 to 4 (severity); I collapse it to presence (1) or absence (0), the standard use of this
set.

## What's in here

- `analysis.ipynb` is the full walkthrough: loading, EDA, modelling, threshold tuning, with
  the code, charts and commentary.
- `build_notebook.py` is the script that builds and executes that notebook (my usual
  workflow, so the outputs in the notebook are real, not pasted).
- `app.py` is a small Streamlit app that loads the trained model for a live risk estimate.
- `figures/` has the saved plots, `data/` has the raw dataset, `model/` has the saved model.

## Approach

A stratified 80/20 split so both classes land in the test set. Features scaled for the
logistic regression, left raw for the random forest. I compared the two, then looked hard
at the decision threshold rather than accepting the default 0.5.

A couple of decisions worth calling out:

- **Logistic regression as the primary model.** The two models are close, and the forest
  edges accuracy, but the logistic regression has the higher AUC here and it is
  interpretable, which matters when someone asks why a patient was flagged. In a clinical
  setting that explainability is worth more than a point of accuracy.
- **Moving the threshold to buy recall.** Because a missed diagnosis is the expensive error,
  I dropped the threshold to 0.35. That lifts recall on the disease class from 0.79 to 0.89
  for a small precision cost.

## Results

On the held-out test set (60 patients, 28 with disease):

| model | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| logistic regression | 0.833 | 0.846 | 0.786 | 0.815 | 0.950 |
| random forest | 0.867 | 0.885 | 0.821 | 0.852 | 0.943 |

Both models are strong, with AUC around 0.95, which is what you expect on this dataset. The
features that carry the signal line up with clinical intuition: chest pain type, number of
major vessels (`ca`), the thalassemia result, exercise induced angina, and ST depression.

Tuning the logistic regression threshold:

| threshold | recall (disease) | precision (disease) |
|---|---|---|
| 0.50 (default) | 0.786 | 0.846 |
| 0.35 (tuned) | 0.893 | 0.833 |

So I catch 25 of 28 disease cases instead of 22, and the precision barely moves. That is the
trade I would make here.

## Running it

```
pip install -r requirements.txt
python build_notebook.py      # trains, builds analysis.ipynb, saves figures and the model
streamlit run app.py          # interactive risk estimate
```

Or just open `analysis.ipynb` to read the executed walkthrough.

## Honest limitations

- This is 297 patients from one study in the late 1980s. It is a classic teaching dataset,
  not current clinical practice, and some feature definitions (like `thal`) are specific to
  how that study was run.
- The test set is only 60 patients, so a single reclassified case moves recall by almost two
  points. I would not over read small gaps between the models.
- A real deployment would need calibration to the local population, a prospective validation
  instead of a random split, and sign off from clinicians. This is a modelling exercise, not
  a medical device, and the app says so.

Data source: Heart Disease dataset, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/45/heart+disease).

## License

MIT, see `LICENSE`.
