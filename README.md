# Employee Salary Prediction — Regression

## Project Overview

This project applies supervised machine learning to predict employee salaries based on professional and demographic features.

The objective is to explore the complete regression workflow, including exploratory data analysis, preprocessing, model comparison, evaluation, and salary prediction for new employee records.

## Dataset

**Source:** [Employee Salary Dataset — GitHub](https://github.com/RobStilson/Employee-Datasets)

The dataset contains approximately **10,000 employee records** with information about:

- Age and years of experience
- Department and job title
- Education level and location
- Employee identifiers and gender
- Salary (target variable)

Employee ID, name, and gender were excluded from the model's input features.

**Note:** The dataset exhibits highly structured patterns and may be synthetically generated. Its salary currency and payment period have not been verified.

## Project Structure

```text
Employee-Salary-Prediction/
├── data_exploration.ipynb
├── model_training.ipynb
├── Employers_data.csv
└── README.md
```

## 1. Exploratory Data Analysis

EDA was performed to understand salary distributions and identify relationships between input features and the target.

**Key findings:**

- Salary distribution is multimodal, suggesting distinct salary bands.
- Experience has a strong positive relationship with salary.
- Job title and education level show clear salary differences.
- Department is associated with salary variation, although distributions overlap.
- Age and experience are highly correlated (r = 0.98), indicating potential multicollinearity.

## 2. Data Preprocessing

The dataset was divided into **80% training and 20% testing**, using a reproducible random split.

Preprocessing was implemented with Scikit-learn's `ColumnTransformer` and `Pipeline`:

- **StandardScaler:** Standardizes numerical features.
- **OneHotEncoder:** Encodes categorical features.
- **Pipeline:** Combines preprocessing and model training while reducing data leakage during cross-validation.

The six input features were Age, Experience_Years, Department, Job_Title, Education_Level, and Location.

## 3. Model Training and Comparison

Five regression approaches were evaluated using **5-Fold Cross-Validation** on the training data:

- Dummy Regressor (Baseline)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Models were compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

### Cross-Validation Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| **Linear Regression** | **3,429.23** | **4,289.89** | **0.9913** |
| Gradient Boosting | 3,479.63 | 4,352.90 | 0.991 |
| Decision Tree | 3,757.06 | 4,663.83 | 0.990 |
| Random Forest | 3,860.71 | 4,854.78 | 0.989 |
| Dummy Baseline | 39,513.09 | 46,101.44 | ~0.000 |

**Linear Regression** achieved the best average cross-validation performance and was selected as the final model.

## 4. Final Model Evaluation

The selected Linear Regression Pipeline was fitted using the complete training dataset and evaluated on the held-out test set.

| Metric | Test Result |
|---|---:|
| MAE | 3,337.97 |
| RMSE | 4,168.12 |
| R² | 0.9918 |

The model demonstrated strong predictive performance, with test results closely matching the cross-validation estimates.

Visual evaluation using Actual vs. Predicted and Residual plots showed predictions close to the ideal line and residuals generally distributed around zero.

## 5. Salary Prediction

An interactive input section was implemented within the training notebook.

Users can enter an employee's:

- Age
- Years of experience
- Department
- Job title
- Education level
- Location

The trained Pipeline processes the inputs and returns an estimated salary.

Unseen categorical values are handled using `OneHotEncoder(handle_unknown="ignore")`, although predictions for unfamiliar categories may be less reliable.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Limitations

- The dataset appears highly structured and may not represent real-world salary patterns.
- Strong correlation between age and experience may affect coefficient interpretation.
- Excluding gender as an input does not guarantee that predictions are free from indirect bias.
- Real-world validation is required before using the model for employment or compensation decisions.

## Conclusion

This project demonstrates an end-to-end regression workflow, from data exploration to model evaluation and prediction.

**Linear Regression outperformed the more complex models**, demonstrating that model complexity does not always translate into better predictive performance.

The final model achieved an **R² of 0.9918** and an **MAE of approximately 3,338** on the held-out test dataset.

These results reflect performance on the available dataset and should not be assumed to generalize to real-world salary data.
