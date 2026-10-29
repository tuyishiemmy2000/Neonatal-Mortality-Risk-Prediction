# Predicting Neonatal Mortality Risk Using Machine Learning and Rwanda DHS Data

**Project:** Rwanda Neonatal Mortality Risk Prediction  
**Focus:** Maternal and newborn health, machine learning, and district-level analysis  
**Country:** Rwanda  
**Data source:** Rwanda Demographic and Health Survey (DHS)  
**Dashboard:** Streamlit  
**Repository:** [Neonatal-Mortality-Risk-Prediction](https://github.com/tuyishiemmy2000/Neonatal-Mortality-Risk-Prediction)

## 1. Project Overview

Neonatal mortality refers to the death of a newborn within the first 28 days of life. It remains an important public health concern, particularly in settings where access to healthcare, socioeconomic conditions, maternal health, and birth-related factors may influence newborn survival.

This project uses Rwanda Demographic and Health Survey (DHS) birth-history data to investigate factors associated with neonatal mortality and develop machine learning models for estimating neonatal mortality risk.

The project explores maternal characteristics, antenatal care, household socioeconomic conditions, healthcare accessibility, and delivery-related information. It also compares observed neonatal mortality indicators and model-predicted risk across districts in Rwanda.

The long-term goal is to explore how data-driven risk assessment could support maternal and newborn health research, public health planning, and the development of screening tools for community and facility health workers.

## 2. Problem Statement

Neonatal mortality is influenced by multiple factors, including maternal characteristics, access to healthcare, socioeconomic circumstances, antenatal care, and conditions surrounding birth.

Understanding these factors and identifying patterns associated with newborn deaths can help researchers and public health planners investigate populations and geographic areas that may require additional attention.

This project addresses the following questions:

- Which maternal, socioeconomic, antenatal, and delivery-related characteristics are associated with neonatal mortality?
- Can machine learning identify newborns with higher predicted neonatal mortality risk using available DHS information?
- How does model performance differ between logistic regression and Random Forest?
- How do observed neonatal mortality indicators and predicted risk vary across Rwanda's districts?
- How can these findings be presented through an interactive dashboard?

## 3. Project Objectives

The main objectives are to:

1. Clean and preprocess Rwanda DHS birth-history data.
2. Construct a neonatal mortality outcome using child survival status and age at death.
3. Engineer features suitable for machine learning.
4. Develop separate prediction models for antenatal assessment and assessment at delivery.
5. Compare Logistic Regression and Random Forest classifiers.
6. Evaluate model performance using metrics appropriate for an imbalanced classification problem.
7. Analyze observed neonatal mortality rates across districts.
8. Examine average predicted neonatal mortality risk by district.
9. Visualize results using charts, tables, and an interactive Streamlit dashboard.
10. Produce exploratory evidence to inform further maternal and newborn health research.

## 4. Dataset

The project uses Rwanda Demographic and Health Survey birth-history information, supplemented by relevant maternal, household, and healthcare-access variables where available.

The analysis uses a prepared child-level dataset created from DHS information. The working dataset has included variables such as `District`, `Child is Alive`, `Age at death category`, and `current_age_of_children`.

### Data source

- [The DHS Program](https://dhsprogram.com/)
- [DHS Program data access](https://dhsprogram.com/data/)

Access to DHS microdata is subject to the program's registration, approval, and data-use requirements. The raw survey files and restricted data are not included in this GitHub repository.

### Main variables

| Variable | Description |
|---|---|
| Mother's Age | Age of the mother |
| Mother's Education | Highest educational level attained |
| Wealth Status | Household wealth category |
| Residence | Urban or rural residence |
| District | District associated with the survey record |
| ANC Visits | Number of antenatal care visits |
| First ANC Timing | Timing of the first antenatal care visit |
| Place of Delivery | Location where delivery occurred |
| Caesarean Delivery | Whether delivery was by caesarean section |
| Previous Caesarean Section | History of a previous caesarean section |
| Birth Weight | Recorded birth weight, where available |
| Sex of Child | Sex of the child |
| Birth Index / Birth Order | Child's position in the mother's birth history |
| Transport to Health Facility | Usual transport method to a health facility |
| Minutes to Facility | Travel time to a health facility |
| Child is Alive | Child survival status |
| Age at Death Category | Recorded age-at-death category |
| Current Age of Child | Age information used to establish survival status when applicable |

Variable availability and coding depend on the DHS files used and the survey documentation. Missing values, special codes, and merged records must be checked before analysis.

## 5. Definition of the Prediction Target

The target variable is named `neonatal_death`.

- **1 — Neonatal death:** The child died during the first 28 days of life.
- **0 — No neonatal death:** The child survived beyond the neonatal period or died after the neonatal period, provided the information establishes that neonatal death did not occur.
- **Excluded — Unknown status:** Records for which the available information cannot establish neonatal mortality status reliably.

The target is derived from child survival status and age-at-death information. DHS age-at-death codes must be interpreted according to the relevant survey documentation.

A child who died at an unknown age must not automatically be classified as a neonatal death or as a non-neonatal death. Similarly, a living child whose age information is insufficient to establish the required eligibility criteria should not automatically be assigned a negative outcome.

Careful target construction is essential because errors in the outcome definition can affect every subsequent modeling result.

## 6. Feature Engineering and Preprocessing

Feature engineering is implemented in `src/build_features.py`.

The process transforms the prepared DHS data into model-ready variables and constructs the neonatal mortality target.

### Main predictor groups

**Antenatal assessment**

The antenatal model uses information intended to be available before delivery, including:

- Mother's age
- Birth order
- Number of antenatal care visits
- Timing of the first antenatal care visit
- Travel time to a health facility
- Usual transport method to a health facility
- Urban or rural residence
- Previous caesarean section
- Smoking status
- Household wealth category
- Mother's education
- District

**Assessment at delivery**

The delivery-stage model includes the antenatal predictors and additional information available at or after delivery, including:

- Birth weight
- Sex of the child
- Caesarean delivery
- Facility delivery

The exact feature lists are defined in the feature-engineering and training scripts.

### Preprocessing steps

- Validate the dataset structure and relevant variables.
- Remove duplicate records where necessary and verify child-level identifiers.
- Handle DHS special codes, including codes representing unknown values.
- Convert variables into appropriate numerical or categorical representations.
- Convert birth weight to a consistent unit where required.
- Construct the binary neonatal mortality outcome.
- Exclude records whose target status cannot be established reliably.
- Impute missing predictor values using training-set statistics within the modeling pipeline.

Missing values and special codes must be handled before model fitting. Information unavailable at the intended prediction time must not be used as a predictor.

**Important distinction:** The antenatal model is intended to estimate risk using information available before delivery. The delivery-stage model can use additional birth and delivery information. These are separate assessments rather than one model applied indiscriminately at both stages.

## 7. Machine Learning Models

The project compares two classification algorithms:

### Logistic Regression

Logistic Regression provides a baseline classification model for estimating the probability of neonatal death. It also supports comparison with a more flexible tree-based algorithm.

### Random Forest

Random Forest combines multiple decision trees to model potentially nonlinear relationships and interactions among predictors.

The project uses class-imbalance handling and decision-threshold selection because neonatal deaths represent a small proportion of eligible birth records.

The main model comparison covers two prediction stages:

| Prediction stage | Logistic Regression | Random Forest |
|---|---|---|
| Before birth | Antenatal assessment | Antenatal assessment |
| At or after delivery | Delivery-stage assessment | Delivery-stage assessment |

The Streamlit dashboard uses the saved Random Forest models for its pregnancy and delivery prediction pages, according to the models and configuration available in the project outputs.

### Decision threshold

The default classification threshold of 0.50 may be unsuitable when neonatal deaths are uncommon. A lower threshold may identify more potential positive cases, but it can also increase false-positive classifications.

The training workflow selects a threshold using training data and cross-validation, based on a target recall. The selected threshold is then applied to the held-out test data.

The threshold is a modeling choice, not a clinically validated cutoff.

## 8. Model Evaluation

The models are evaluated using multiple metrics because accuracy alone can be misleading when the outcome is highly imbalanced.

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct classifications |
| Precision | Proportion of predicted neonatal deaths that are actual neonatal deaths |
| Recall (Sensitivity) | Proportion of actual neonatal deaths detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases across thresholds |
| PR-AUC | Precision-recall performance, particularly relevant for rare outcomes |
| Confusion Matrix | Counts of true positives, true negatives, false positives, and false negatives |

The model comparison should consider both the ability to identify neonatal deaths and the number of records classified as high risk.

Results depend on the eligible sample, target definition, train/test split, feature availability, and decision threshold. The reported results should be reproduced using the current training script rather than assumed from earlier experiments.

## 9. District-Level Analysis

The project investigates geographic differences in observed neonatal mortality and model-predicted risk across Rwanda's districts.

The district analysis includes:

- Number of eligible birth records
- Number of observed neonatal deaths
- Observed neonatal deaths per 1,000 eligible births
- Mean predicted neonatal mortality risk
- Percentage of records classified into a relative high-risk group
- District comparisons and rankings


### Predicted neonatal mortality risk

Mean predicted risk is calculated from the model's estimated probabilities for eligible records in each district.

Predicted risk is different from the observed neonatal mortality rate. It reflects the model's estimates and is not a count of confirmed deaths.

Districts may also be grouped into relative risk categories for visualization. Such categories describe comparisons within the analyzed dataset and should not be interpreted as official clinical or public health classifications.

### Interpretation and limitations

District-level comparisons are exploratory. DHS sampling design, survey weights, clustering, small numbers of observed deaths, and uncertainty in district-level estimates can affect interpretation.

Additional survey-aware estimation and external validation are needed before these results can support formal district comparisons or resource-allocation decisions.

## 10. Interactive Streamlit Dashboard

The project includes a Streamlit dashboard designed to present the analysis in an accessible format.

The dashboard navigation includes:

1. **National overview** — summary indicators and national-level findings.
2. **Pregnancy prediction** — estimates risk using information intended to be available before delivery.
3. **Delivery prediction** — estimates risk using antenatal and delivery-stage information.
4. **District analysis** — explores observed mortality indicators and average predicted risk by district.
5. **Model information** — explains the modeling approach, evaluation metrics, and limitations.

Dashboard content depends on the model files and output datasets generated by the training workflow.

The dashboard is intended to support exploration and demonstration of the analysis, not to replace professional medical assessment or established clinical protocols.

## 11. Project Structure

```text
Neonatal-Mortality-Risk-Prediction/
│
├── data/
│   └── merged_table.csv
│       # Prepared input data; not committed to GitHub
│
├── src/
│   ├── build_features.py
│   └── train_model.py
│
├── app/
│   └── app2.py
│
├── notebooks/
│   └── # Exploratory analysis and experiments
│
├── outputs/
│   ├── models/
│   │   ├── pregnancy_model.joblib
│   │   └── delivery_model.joblib
│   ├── model_thresholds.json
│   ├── district_risk.csv
│   ├── pregnancy_district_risk.csv
│   ├── delivery_district_risk.csv
│   ├── national_risk_summary.csv
│   └── # Other generated evaluation outputs
│
├── requirements.txt
├── .gitignore
└── README.md
```

The exact output filenames may vary with the current training-script configuration. The tree above describes the intended organization; only files actually generated or included in the repository will be available.

### Main files

- `src/build_features.py` — prepares model features and constructs the prediction target.
- `src/train_model.py` — trains and evaluates models, selects thresholds, and generates analysis outputs.
- `app/app.py` — runs the interactive dashboard.
- `data/` — contains prepared input data, which is excluded from the public repository.
- `outputs/` — contains generated model artifacts, thresholds, evaluation summaries, and district-level results.
- `notebooks/` — contains exploratory analysis and experiments.

## 12. Technologies Used

- **Python** — core programming language
- **Pandas** — data cleaning and manipulation
- **NumPy** — numerical computation
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Matplotlib / Altair** — data visualization
- **Streamlit** — interactive dashboard
- **Jupyter Notebook** — exploratory data analysis and experimentation
- **Git and GitHub** — version control and project sharing



## 14. Expected Outputs

The project is designed to generate:

- A cleaned, feature-engineered dataset
- Trained Logistic Regression and Random Forest models for the two assessment stages
- Model evaluation metrics and confusion matrices
- Selected classification thresholds
- Feature-importance summaries where supported
- District-level observed neonatal mortality summaries
- District-level average predicted risk summaries
- Visual comparisons of district results
- An interactive Streamlit dashboard

Actual outputs depend on the current scripts, data availability, and successful execution of the workflow.

## 15. Limitations and Responsible Use

This project is an exploratory machine learning study using survey data. Its results should be interpreted with care.

Key limitations include:

- **Rare outcome:** Neonatal deaths form a small proportion of eligible records, making reliable detection challenging.
- **Missing information:** Some predictors, including birth weight, may be missing or inconsistently recorded.
- **Survey design:** DHS observations arise from a complex sampling design that must be considered when estimating population-level indicators.
- **District-level uncertainty:** Small numbers of deaths can make district comparisons unstable.
- **Prediction timing:** Delivery-stage variables must not be used in a model intended to operate before delivery.
- **Generalizability:** Performance on the analyzed dataset does not guarantee performance in other populations, facilities, or future periods.
- **Clinical validity:** The models have not been established as validated clinical decision tools solely through this analysis.

The dashboard should be used for research, demonstration, and exploratory risk assessment. Its predictions must not replace clinical judgment, professional assessment, or established referral and care protocols.

## 16. Future Improvements

Potential future work includes:

- Incorporating DHS survey weights and complex-survey methods into descriptive analyses.
- Improving target validation and checking child-level record linkage.
- Evaluating probability calibration and uncertainty.
- Conducting more robust validation and external testing.
- Comparing additional approaches where scientifically justified.
- Improving interpretability and documenting feature contributions.
- Assessing fairness and performance across relevant population subgroups.
- Refining dashboard accessibility for researchers and health workers.
- Working with maternal and newborn health experts to assess whether the approach is suitable for further research and validation.

## 17. Data Source and Acknowledgment

This project uses Rwanda Demographic and Health Survey data, subject to the applicable DHS data-access terms and permissions.

Researchers using the data should cite the relevant Rwanda DHS report, survey documentation, and implementing institutions according to the official DHS citation requirements.

**Data source:** [The DHS Program](https://dhsprogram.com/)

## 18. Author and Repository

**GitHub repository:**  
https://github.com/tuyishiemmy2000/Neonatal-Mortality-Risk-Prediction

This project demonstrates the application of data preparation, machine learning, model evaluation, geographic analysis, and interactive visualization to an important maternal and newborn health research question in Rwanda.

## 18. AI Disclosure

Artificial intelligence (AI) tools were used as supportive resources during the development of this project. Their assistance included code development and debugging, Designing the structure of my project

The project design, data preparation, model implementation, evaluation, interpretation of results, and final decisions remain the responsibility of the project author. AI-generated suggestions were reviewed and adapted to the project's requirements.

The machine learning models were trained using Rwanda DHS data, and their outputs depend on the data, preprocessing procedures, features, and evaluation methods used. AI assistance does not establish the scientific validity or clinical reliability of the resulting predictions.

The project does not use AI-generated outputs as a substitute for evidence-based clinical assessment. All findings should be interpreted in light of the study's methodological limitations and the need for further validation.

