# Rwanda Neonatal Mortality Risk Prediction

### Predicting Neonatal Mortality Risk Using Machine Learning and Rwanda DHS Data

## 1. Project Overview

Neonatal mortality refers to the death of a newborn within the first 28 days of life. It remains an important public health concern because early identification of newborns at increased risk can help health workers prioritize care and follow-up.

This project uses data from the Rwanda Demographic and Health Survey (DHS) to explore factors associated with neonatal mortality and develop a machine learning model to predict neonatal death risk.

The project uses a Random Forest classifier and compares observed neonatal mortality and predicted risk across districts in Rwanda.

The long-term goal is to support evidence-based maternal and newborn health planning and inform the development of a risk-screening tool for community and facility health workers.

## 2. Problem Statement

Some mothers and newborns may face increased risks due to socioeconomic conditions, limited access to healthcare, inadequate antenatal care, and pregnancy or delivery-related factors.

Identifying patterns associated with neonatal mortality can help researchers and public health planners understand where additional support may be needed.

This project investigates how machine learning can use available maternal, birth, and healthcare-access information to predict neonatal mortality risk and examine differences between districts in Rwanda.

## 3. Project Objectives

The main objectives are to:

* Analyze maternal and birth-related factors associated with neonatal mortality.
* Clean and preprocess Rwanda DHS birth-history data.
* Engineer relevant features for machine learning.
* Develop a Random Forest classification model.
* Evaluate model performance using appropriate classification metrics.
* Compare observed neonatal mortality rates across districts.
* Examine predicted neonatal mortality risk by district.
* Visualize results using tables and charts.
* Provide exploratory evidence to support maternal and newborn health research.

## 4. Dataset

The project uses Rwanda Demographic and Health Survey birth-history data.

**Dataset source:** Rwanda DHS / DHS Program.

The data include information about births, maternal characteristics, household socioeconomic conditions, healthcare utilization, and child survival.

The dataset used for analysis is a cleaned CSV file named `processed_data_cleaned.csv`.

### Main variables

| Variable                  | Description                                |
| ------------------------- | ------------------------------------------ |
| Mother's Age              | Age of the mother                          |
| Highest Educational Level | Mother's education level                   |
| Wealth Status             | Household wealth category                  |
| Place of Residence        | Urban or rural residence                   |
| District                  | District associated with the survey record |
| ANC Visits                | Number of antenatal care visits            |
| First ANC Timing          | Timing of the first antenatal care visit   |
| Place of Delivery         | Location where delivery occurred           |
| Caesarean Delivery        | Whether delivery was by caesarean section  |
| Birth Weight              | Recorded birth weight, where available     |
| Birth Index               | DHS birth-history index                    |
| Child is Alive            | Child survival status                      |
| Age at Death Category     | Recorded age-at-death category             |

Variable availability and coding depend on the specific DHS files and survey documentation. Missing values and special DHS codes must be checked before analysis.

## 5. Definition of the Prediction Target

The target variable is `neonatal_death`.

* `1` — Death occurred during the first 28 days of life.
* `0` — No neonatal death, including surviving children and children whose deaths occurred after the neonatal period.

Records with unknown or unusable death ages must be handled carefully because their neonatal mortality status cannot be established reliably.

The target is constructed from child survival status and age-at-death information.

**Important:** The DHS age-at-death coding must be verified against the survey documentation. The outcome should not classify every child death as a neonatal death.

## 6. Feature Engineering

Feature engineering transforms existing variables into useful inputs for machine learning (`src/build_features.py`).

The model uses these features: mother's age, birth order, number of ANC visits, timing of first ANC visit, birth weight, minutes to the nearest health facility, child's sex, urban or rural residence, caesarean delivery, previous caesarean, smoking, facility delivery, walking as the usual transport to a facility, wealth (1-5), mother's education (0-3) and district.

Other steps in `build_features.py`:

* Duplicate rows created by the merge are removed (one row per child).
* DHS "don't know" codes (birth weight 9998, ANC visits 98, first ANC timing 98) are set to missing.
* Missing values are filled with the training-set median inside the model pipeline.
* Children who died at an unknown age are excluded because their neonatal status cannot be established.

**Important - prediction time:** the current model includes birth weight, delivery location and delivery mode, so it is a risk-identification model for use at or after delivery. An antenatal screening model should use only information available before those events, so those features would need to be removed.

## 7. Model

* **Algorithm:** Random Forest (300 trees, `min_samples_leaf=5`, `class_weight="balanced"`).
* **Split:** 80% training and 20% testing, stratified by outcome.
* **Cutoff:** neonatal deaths are rare, so the default 0.5 cutoff catches almost none. The cutoff is chosen from cross-validated predictions on the training data to reach a target recall (default 60%), and the model is tested once on the test set.
* **District risk:** each child receives a cross-validated predicted risk. Districts are ranked by mean predicted risk and grouped into Low, Medium and High thirds.

## 8. Model Evaluation

The model is evaluated using several metrics:

| Metric               | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| Accuracy             | Proportion of predictions that are correct                              |
| Precision            | Proportion of predicted neonatal deaths that are actual neonatal deaths |
| Recall (Sensitivity) | Proportion of actual neonatal deaths detected                           |
| F1-score             | Balance between precision and recall                                    |
| ROC-AUC              | Measures the model's ability to rank cases by risk                      |
| Confusion Matrix     | Shows correct and incorrect classifications                             |

Accuracy alone is insufficient when neonatal deaths are uncommon. Recall, precision, calibration, and false-negative rates are particularly important for a potential screening application.

Performance values depend on the chosen cutoff and the random split. Run `src/train_model.py` to reproduce them; with only a few hundred neonatal deaths in the survey, results can vary between splits.

## 9. District-by-District Comparison

An important component of this project is comparing neonatal mortality indicators across districts in Rwanda.

The district analysis includes:

* Total eligible births per district.
* Number of observed neonatal deaths.
* Observed neonatal deaths per 1,000 eligible births.
* Mean predicted neonatal mortality risk.
* Percentage of records classified into a relative high-risk group.

The observed mortality rate is calculated as:

**Neonatal mortality rate per 1,000 births**

District comparisons can help identify geographic patterns and generate hypotheses for further investigation.

However, DHS survey estimates may be uncertain at district level because of sampling design and small numbers of observed deaths. These comparisons should be treated as exploratory unless appropriate survey weighting, uncertainty estimation, and validation are performed.

## 10. Project Structure

```text
Neonatal-Mortality-Risk-Prediction/
│
├── data/
│   └── processed_data_cleaned.csv     (not on GitHub - see Data Source)
│
├── src/
│   ├── build_features.py              (features and target)
│   └── train_model.py                 (model, evaluation, district results)
│
├── app/
│   └── app.py                         (optional Streamlit dashboard)
│
├── notebooks/                         (exploratory analysis and experiments)
│
├── outputs/
│   ├── district_risk.csv              (district-level results)
│   ├── district_risk.png              (district risk chart)
│   └── feature_importance.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

* `data/` - input data. The DHS data file is **not** included in the repository.
* `src/build_features.py` - creates engineered features and the modelling target (`neonatal_death`).
* `src/train_model.py` - trains the Random Forest, chooses the cutoff, evaluates the model and writes district results to `outputs/`.
* `app/app.py` - optional Streamlit dashboard.
* `notebooks/` - exploratory data analysis and model experiments.
* `outputs/` - results produced by `train_model.py`.
* `requirements.txt` - Python dependencies.

## 11. Technologies Used

* Python— programming language.
* Pandas — data cleaning and manipulation.
* NumPy — numerical operations.
* Scikit-learn — preprocessing, Random Forest, and model evaluation.
* Matplotlib — data visualization.
* Streamlit — optional interactive dashboard.
* Jupyter Notebook — exploratory analysis and experimentation.
* Git and GitHub — version control and project collaboration.

## 12. Expected Outputs

The project aims to produce:

* A cleaned and feature-engineered dataset.
* A trained Random Forest model.
* Model performance metrics.
* A feature-importance summary.
* District-level neonatal mortality comparison tables.
* Charts comparing observed mortality rates across districts.
* District-level summaries of predicted risk.
* An optional interactive dashboard.

## 13. How to Run

```bash
pip install -r requirements.txt
```

1. Place `processed_data_cleaned.csv` in the `data/` folder. The DHS data are not included in this repository; request access from the DHS Program (dhsprogram.com).
2. Train and evaluate the model from the project root:

```bash
python src/train_model.py
```

3. Optional dashboard:

```bash
streamlit run app/app.py
```

## 14. Data Source and Acknowledgment

The analysis uses Rwanda Demographic and Health Survey data, subject to the applicable DHS data-access terms and permissions.

Researchers should acknowledge the Rwanda DHS implementing institutions and the DHS Program in accordance with the survey's official citation requirements.

