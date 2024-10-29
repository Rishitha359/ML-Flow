import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
 
# Set the MLflow tracking URI if you have a specific server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
 
# Create a new experiment or set an existing one
experiment_name = "test_experiment"  # Change to a simple name
mlflow.set_experiment(experiment_name)
 
# Load your data
final_df = pd.read_csv(r'C:\Users\VenkataRishitha\Training\Final Project 30-09-2024\Data Science\final_df.csv')  # Load your data here
 
# Features and target variable
X = final_df.drop('is_promoted', axis=1)
y = final_df['is_promoted']
 
# Convert string values to numeric
X = X.apply(pd.to_numeric, errors='coerce')
 
# Drop rows with missing values
X.dropna(inplace=True)
y = y[X.index]
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define classifiers with parameters
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {'C': 1.0, 'solver': 'lbfgs'}
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': 100, 'max_depth': None}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {'max_depth': None, 'min_samples_split': 2}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(),
        'params': {'n_estimators': 100, 'learning_rate': 0.1}
    },
    'Support Vector Machine': {
        'model': SVC(probability=True),
        'params': {'C': 1.0, 'kernel': 'rbf'}
    }
}
 
# Start a new MLflow run for the experiment
for model_name, model_info in models.items():
    model = model_info['model']
    params = model_info['params']
   
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\nModel: {model_name}")
 
        # Log model parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
 
        # Fit the model
        model.fit(X_train, y_train)
 
        # Predict probabilities for both train and test sets
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
 
        roc_auc_train = roc_auc_score(y_train, y_prob_train)
        roc_auc_test = roc_auc_score(y_test, y_prob_test)
 
        # Log metrics to MLflow
        mlflow.log_metric("roc_auc_train", roc_auc_train)
        mlflow.log_metric("roc_auc_test", roc_auc_test)
 
        # Compute ROC curve to find the optimal threshold
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_prob_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob_test)
 
        # Find the optimal threshold
        optimal_idx_train = (tpr_train - fpr_train).argmax()
        optimal_threshold_train = thresholds_train[optimal_idx_train]
 
        optimal_idx_test = (tpr_test - fpr_test).argmax()
        optimal_threshold_test = optimal_threshold_train
 
        print(f'Optimal Threshold (Train): {optimal_threshold_train}')
        print(f'Optimal Threshold (Test): {optimal_threshold_test}')
 
        # Convert probabilities to binary predictions
        y_pred_train = (y_prob_train >= optimal_threshold_train).astype(int)
        y_pred_test = (y_prob_test >= optimal_threshold_test).astype(int)
 
        # Calculate F1 scores for both train and test sets
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
 
        # Log F1 scores to MLflow
        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)
 
        print(f'F1 Score (Train): {f1_train}')
        print(f'F1 Score (Test): {f1_test}')
 
        # Log the model
        mlflow.sklearn.log_model(model, model_name)  # Use the model name directly
 
# No need for an explicit end run since each run is handled by 'with' context