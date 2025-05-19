# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import shap
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Load the data (replace with your actual data loading code)
# Assuming your data is in a CSV file with a column 'loan_taken' (1 for taken, 0 for not taken)
def load_data(file_path='your_esg_finance_data.csv'):
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {data.shape}")
    print(data.head())
    print(data.info())
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values per column:")
    print(missing_values[missing_values > 0])
    
    return data

def preprocess_data(data):
    # Separate features and target
    X = data.drop('loan_taken', axis=1)  # Assuming 'loan_taken' is your target column
    y = data['loan_taken']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    
    # Create preprocessing pipelines
    # For numerical features: impute missing values and apply bucketing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('bucketing', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'))
    ])
    
    # For categorical features: impute missing values and apply one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Create the full pipeline with preprocessor and random forest classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test, model, preprocessor

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return model

def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    output_features = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe.named_steps['bucketing'], 'get_feature_names_out'):
                if name == 'num':
                    current_features = pipe.named_steps['bucketing'].get_feature_names_out(features)
                else:
                    current_features = pipe.named_steps['onehot'].get_feature_names_out(features)
                output_features.extend(current_features)
            else:
                output_features.extend(features)
    
    return np.array(output_features)

def analyze_feature_importance(model, X_test, y_test):
    # Get feature names after preprocessing
    feature_names = get_feature_names(model.named_steps['preprocessor'])
    
    # Get feature importances from the random forest
    importances = model.named_steps['classifier'].feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Method 2: Permutation Importance (more reliable for feature importance)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importances')
    plt.bar(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
    plt.xticks(range(len(sorted_idx)), feature_names[sorted_idx], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Method 3: SHAP values for more detailed feature importance
    # Create a smaller dataset for SHAP analysis (for computational efficiency)
    X_shap = X_test.iloc[:100]  # Use a subset for SHAP analysis
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(model.named_steps['preprocessor'].transform(X_shap))
    
    # Plot SHAP summary
    shap.summary_plot(shap_values[1], model.named_steps['preprocessor'].transform(X_shap), 
                     feature_names=feature_names)

def perform_clustering_analysis(X_train, y_train, model):
    # Apply K-means clustering on the preprocessed data
    # First, transform the training data
    X_train_transformed = model.named_steps['preprocessor'].transform(X_train)
    
    # Determine optimal number of clusters using elbow method
    inertia = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train_transformed)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    # Choose optimal k (let's say k=4 based on the elbow curve)
    optimal_k = 4  # Adjust based on the elbow curve
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train_transformed)
    
    # Add cluster labels to the training data
    X_train_with_clusters = X_train.copy()
    X_train_with_clusters['cluster'] = cluster_labels
    
    # Analyze loan distribution within each cluster
    cluster_loan_distribution = pd.DataFrame({
        'cluster': cluster_labels,
        'loan_taken': y_train.values
    }).groupby('cluster')['loan_taken'].agg(['mean', 'count'])
    
    print("Loan distribution by cluster:")
    print(cluster_loan_distribution)
    
    # Visualize clusters (using PCA for dimensionality reduction)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_transformed)
    
    plt.figure(figsize=(10, 8))
    for i in range(optimal_k):
        plt.scatter(X_train_pca[cluster_labels == i, 0], X_train_pca[cluster_labels == i, 1], 
                    label=f'Cluster {i}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='red', marker='X', label='Centroids')
    plt.legend()
    plt.title('Clusters of Companies (PCA-reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    
    return X_train_transformed, cluster_labels, optimal_k

def build_cluster_models(X_train, y_train, cluster_labels, optimal_k, preprocessor):
    # Build separate models for each cluster
    cluster_models = {}
    for cluster in range(optimal_k):
        # Get data for this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        X_cluster = X_train.iloc[cluster_indices]
        y_cluster = y_train.iloc[cluster_indices]
        
        # Create and train a model for this cluster
        cluster_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Only train if we have enough samples
        if len(X_cluster) > 20:  # Arbitrary threshold
            cluster_model.fit(X_cluster, y_cluster)
            cluster_models[cluster] = cluster_model
            print(f"Model for Cluster {cluster} trained with {len(X_cluster)} samples")
        else:
            print(f"Not enough samples in Cluster {cluster} ({len(X_cluster)} samples)")
    
    return cluster_models

def tune_hyperparameters(X_train, y_train, model):
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Return best model
    return grid_search.best_estimator_

def identify_likely_loan_takers(X, best_model, data):
    # Predict probabilities for all companies
    all_companies_proba = best_model.predict_proba(X)[:, 1]
    
    # Create a DataFrame with company information and loan probabilities
    company_predictions = pd.DataFrame({
        'company_id': data.index,  # Assuming index contains company identifiers
        'loan_probability': all_companies_proba
    })
    
    # Sort by probability in descending order
    company_predictions = company_predictions.sort_values('loan_probability', ascending=False)
    
    # Get top 50 companies most likely to take loans
    top_companies = company_predictions.head(50)
    print("Top 50 Companies Most Likely to Take Loans:")
    print(top_companies)
    
    # Visualize probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_companies_proba, bins=20, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Probability of Taking a Loan')
    plt.ylabel('Number of Companies')
    plt.title('Distribution of Loan Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Save results to CSV
    company_predictions.to_csv('loan_prediction_results.csv', index=False)
    
    return company_predictions

def save_model(best_model):
    # Save the final model
    joblib.dump(best_model, 'loan_prediction_model.pkl')

def predict_loan_probability(new_data, best_model, X):
    """
    Predict the probability of a company taking a loan.
    
    Parameters:
    new_data (DataFrame): DataFrame containing company features
    
    Returns:
    DataFrame: Original data with added loan probability column
    """
    # Ensure new_data has the same columns as training data
    required_cols = X.columns
    for col in required_cols:
        if col not in new_data.columns:
            new_data[col] = np.nan
    
    # Select only the columns used during training
    new_data = new_data[required_cols]
    
    # Make predictions
    probabilities = best_model.predict_proba(new_data)[:, 1]
    
    # Add predictions to the data
    result = new_data.copy()
    result['loan_probability'] = probabilities
    
    return result

def main():
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y, X_train, X_test, y_train, y_test, model, preprocessor = preprocess_data(data)
    
    # Train and evaluate initial model
    trained_model = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)
    
    # Analyze feature importance
    analyze_feature_importance(trained_model, X_test, y_test)
    
    # Perform clustering analysis
    X_train_transformed, cluster_labels, optimal_k = perform_clustering_analysis(X_train, y_train, trained_model)
    
    # Build cluster-specific models
    cluster_models = build_cluster_models(X_train, y_train, cluster_labels, optimal_k, preprocessor)
    
    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train, y_train, trained_model)
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    print("Best Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Identify companies most likely to take loans
    company_predictions = identify_likely_loan_takers(X, best_model, data)
    
    # Save the model
    save_model(best_model)
    
    return best_model, company_predictions

if __name__ == "__main__":
    best_model, company_predictions = main()
