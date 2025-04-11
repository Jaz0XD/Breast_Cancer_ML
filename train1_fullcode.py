# %% [markdown]
# ## Breast Cancer Survival Prediction Project

# %% [markdown]
# ### 1. Environment Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### 2. Data Loading & Initial Processing
# Load dataset
df = pd.read_excel('Breast-Cancer-METABRIC.xlsx', sheet_name='Sheet1')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing values
df = df.dropna(subset=['overall_survival_months', 'overall_survival_status'])
df['er_status'] = df['er_status'].replace({'Positve': 'Positive'})
df['her2_status'] = df['her2_status'].replace({'Positve': 'Positive'})

# %% [markdown]
# ### 3. Target Variable Engineering
# Create 10-year mortality target
df['10_year_mortality'] = np.where(
    (df['overall_survival_months'] <= 120) & 
    (df['overall_survival_status'] == 'Deceased'), 1, 0)

# Handle right-censored data
df.loc[(df['overall_survival_months'] < 120) & 
       (df['overall_survival_status'] == 'Living'), '10_year_mortality'] = np.nan
df = df.dropna(subset=['10_year_mortality'])

# %% [markdown]
# ### 4. Exploratory Data Analysis
# Survival distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['overall_survival_months'], bins=30, kde=True)
plt.title('Distribution of Survival Time (Months)')
plt.show()

# Kaplan-Meier analysis
kmf = KaplanMeierFitter()
kmf.fit(df['overall_survival_months'], event_observed=df['10_year_mortality'])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.ylabel('Survival Probability')
plt.show()

# %% [markdown]
# ### 5. Feature Engineering
# Select relevant features
features = ['age_at_diagnosis', 'type_of_breast_surgery', 'cancer_type_detailed',
            'er_status', 'her2_status', 'tumor_stage', 'tumor_size', 
            'neoplasm_histologic_grade', 'lymph_nodes_examined_positive',
            'nottingham_prognostic_index', 'chemotherapy', 'hormone_therapy',
            'radio_therapy']

# Preprocessing pipeline
numeric_features = ['age_at_diagnosis', 'tumor_size', 
                    'lymph_nodes_examined_positive', 'nottingham_prognostic_index']
categorical_features = ['type_of_breast_surgery', 'er_status', 'her2_status',
                        'neoplasm_histologic_grade', 'chemotherapy', 
                        'hormone_therapy', 'radio_therapy']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# %% [markdown]
# ### 6. Model Development
X = df[features]
y = df['10_year_mortality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Model pipeline
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Model training and evaluation
results = {}
for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(preprocessor.transform(X_test))
    y_proba = model.predict_proba(preprocessor.transform(X_test))[:,1]
    
    results[name] = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred)
    }

# %% [markdown]
# ### 7. Model Evaluation
# ROC Curve comparison
plt.figure(figsize=(10, 6))
for name, model in models.items():
    RocCurveDisplay.from_estimator(
        model, 
        preprocessor.transform(X_test), 
        y_test, 
        name=name
    )
plt.title('ROC Curve Comparison')
plt.show()

# Performance metrics
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(metrics['classification_report'])
    print("\n" + "="*50 + "\n")

# %% [markdown]
# ### 8. Feature Importance Analysis
# Example for Random Forest
rf = models['Random Forest']
feature_names = (preprocessor.named_transformers_['cat']
                 .get_feature_names_out(categorical_features)).tolist()
feature_names = numeric_features + feature_names

importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importances.head(15))
plt.title('Top 15 Important Features (Random Forest)')
plt.show()
