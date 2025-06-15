import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import load_elliptic_dataset, split_train_test
from models import supervised_model
from utils import (
    evaluate_performance, 
    average_performance_per_timestep, 
    plot_performance_per_timestep,
    calc_occurences_per_timestep
)
from sklearn.metrics import (
    cohen_kappa_score, 
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('always')

st.set_page_config(
    page_title="Bitcoin Fraud Detection",
    page_icon="₿",
    layout="wide"
)

st.title("Bitcoin Fraud Detection using Machine Learning")
st.markdown("""
This application demonstrates the use of machine learning models to detect fraudulent Bitcoin transactions.
The models are trained on the Elliptic dataset, which contains labeled Bitcoin transactions.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page:", ["Model Training", "Data Analysis"])

# Load data
@st.cache_data
def load_data():
    X_df, y_df, edges_df = load_elliptic_dataset()
    X_train, X_test, y_train, y_test = split_train_test(X_df, y_df)
    y_test = y_test.to_frame()
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, y_df, edges_df

# Load and process data
with st.spinner('Loading data...'):
    X_train, X_test, y_train, y_test, y_df, edges_df = load_data()

if page == "Data Analysis":
    st.header("Data Analysis")
    
    st.subheader("Feature Correlation Heatmap")
    st.write("This heatmap shows the correlation between different features in the dataset.")
    
    # Calculate correlation matrix
    with st.spinner('Calculating correlation matrix...'):
        feature_cols = X_train.columns[2:]  # Exclude txId and time_step
        corr_matrix = X_train[feature_cols].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={"shrink": .5},
                    ax=ax)
        
        plt.title('Feature Correlation Heatmap', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)

else:  # Model Training page
    st.header("Model Training")
    
    # Model Selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model to run:",
        ["Random Forest", "XGBoost", "Compare Both"]
    )

    # Hyperparameter Tuning Section
    st.sidebar.header("Hyperparameter Tuning")

    if model_choice in ["Random Forest", "Compare Both"]:
        st.sidebar.subheader("Random Forest Parameters")
        rf_n_estimators = st.sidebar.slider(
            "Number of Trees",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Number of trees in the forest"
        )
        rf_max_depth = st.sidebar.slider(
            "Max Depth",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum depth of the trees"
        )
        rf_min_samples_split = st.sidebar.slider(
            "Min Samples Split",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum number of samples required to split an internal node"
        )
        rf_min_samples_leaf = st.sidebar.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=10,
            value=1,
            help="Minimum number of samples required to be at a leaf node"
        )
        rf_class_weight = st.sidebar.selectbox(
            "Class Weight",
            options=["balanced", "balanced_subsample", None],
            help="Weights associated with classes"
        )

    if model_choice in ["XGBoost", "Compare Both"]:
        st.sidebar.subheader("XGBoost Parameters")
        xgb_n_estimators = st.sidebar.slider(
            "Number of Trees",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Number of gradient boosted trees"
        )
        xgb_learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Step size shrinkage used in update"
        )
        xgb_max_depth = st.sidebar.slider(
            "Max Depth",
            min_value=1,
            max_value=20,
            value=6,
            help="Maximum depth of a tree"
        )
        xgb_subsample = st.sidebar.slider(
            "Subsample",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Subsample ratio of the training instances"
        )
        xgb_colsample_bytree = st.sidebar.slider(
            "Column Sample by Tree",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Subsample ratio of columns when constructing each tree"
        )

    # Display dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Transactions", len(y_df))
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
    with col2:
        st.metric("Fraudulent Transactions", sum(y_df == 1))
        st.metric("Legitimate Transactions", sum(y_df == 0))

    # Run selected model
    if st.sidebar.button("Run Model"):
        with st.spinner('Training and evaluating model...'):
            if model_choice in ["Random Forest", "Compare Both"]:
                st.subheader("Random Forest Results")
                # Create Random Forest model with custom parameters
                rf_params = {
                    'n_estimators': rf_n_estimators,
                    'max_depth': rf_max_depth,
                    'min_samples_split': rf_min_samples_split,
                    'min_samples_leaf': rf_min_samples_leaf,
                    'class_weight': rf_class_weight,
                    'random_state': 42
                }
                model_RF = supervised_model(X_train, y_train, model='RandomForest', **rf_params)
                pred_RF = model_RF.predict(X_test)
                pred_proba_RF = model_RF.predict_proba(X_test)[:, 1]
                
                # Display used parameters
                st.write("Model Parameters:")
                st.json(rf_params)
                
                # Feature Importance
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model_RF.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
                ax.set_title('Top 10 Feature Importance (Random Forest)')
                st.pyplot(fig)
                plt.close(fig)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, pred_RF)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix (Random Forest)')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close(fig)
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, pred_proba_RF)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve (Random Forest)')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, pred_proba_RF)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve (Random Forest)')
                st.pyplot(fig)
                plt.close(fig)
                
                # Performance Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", f"{evaluate_performance(y_test, pred_RF, metric='precision'):.3f}")
                    st.metric("Recall", f"{evaluate_performance(y_test, pred_RF, metric='recall'):.3f}")
                with col2:
                    st.metric("F1 Score", f"{evaluate_performance(y_test, pred_RF):.3f}")
                    st.metric("F1 Micro", f"{evaluate_performance(y_test, pred_RF, metric='f1_micro'):.3f}")
                with col3:
                    st.metric("Cohen Kappa", f"{cohen_kappa_score(y_test, pred_RF):.3f}")
                    st.metric("Matthews Correlation", f"{matthews_corrcoef(y_test, pred_RF):.3f}")

            if model_choice in ["XGBoost", "Compare Both"]:
                st.subheader("XGBoost Results")
                # Create XGBoost model with custom parameters
                xgb_params = {
                    'n_estimators': xgb_n_estimators,
                    'learning_rate': xgb_learning_rate,
                    'max_depth': xgb_max_depth,
                    'subsample': xgb_subsample,
                    'colsample_bytree': xgb_colsample_bytree,
                    'random_state': 42
                }
                model_XGB = supervised_model(X_train, y_train, model='XGBoost', **xgb_params)
                pred_XGB = model_XGB.predict(X_test)
                pred_proba_XGB = model_XGB.predict_proba(X_test)[:, 1]
                
                # Display used parameters
                st.write("Model Parameters:")
                st.json(xgb_params)
                
                # Feature Importance
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model_XGB.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
                ax.set_title('Top 10 Feature Importance (XGBoost)')
                st.pyplot(fig)
                plt.close(fig)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, pred_XGB)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix (XGBoost)')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close(fig)
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, pred_proba_XGB)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve (XGBoost)')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test, pred_proba_XGB)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve (XGBoost)')
                st.pyplot(fig)
                plt.close(fig)
                
                # Performance Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", f"{evaluate_performance(y_test, pred_XGB, metric='precision'):.3f}")
                    st.metric("Recall", f"{evaluate_performance(y_test, pred_XGB, metric='recall'):.3f}")
                with col2:
                    st.metric("F1 Score", f"{evaluate_performance(y_test, pred_XGB):.3f}")
                    st.metric("F1 Micro", f"{evaluate_performance(y_test, pred_XGB, metric='f1_micro'):.3f}")
                with col3:
                    st.metric("Cohen Kappa", f"{cohen_kappa_score(y_test, pred_XGB):.3f}")
                    st.metric("Matthews Correlation", f"{matthews_corrcoef(y_test, pred_XGB):.3f}")

            # Plot performance over time
            if model_choice == "Compare Both":
                st.subheader("Model Performance Over Time")
                f1_RF_timestep = average_performance_per_timestep(X_test, y_test, pred_RF)
                f1_XGB_timestep = average_performance_per_timestep(X_test, y_test, pred_XGB)
                
                model_f1_ts_dict = {'XGBoost': f1_XGB_timestep, 'Random Forest': f1_RF_timestep}
                
                # Create the plot
                occ = calc_occurences_per_timestep()
                illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]
                timesteps = illicit_per_timestep['time_step'].unique()
                
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax2 = ax1.twinx()
                
                # Plot model performance
                ax1.plot(timesteps, f1_RF_timestep, label='Random Forest', linestyle='solid', color='#011f4b', linewidth=3.5)
                ax1.plot(timesteps, f1_XGB_timestep, label='XGBoost', linestyle='dotted', color='#eda84c', linewidth=3.5)
                
                # Plot number of illicit transactions
                ax2.bar(timesteps, illicit_per_timestep['occurences'], color='#0C2D48', alpha=0.3, label='# illicit')
                
                # Customize the plot
                ax1.set_xlabel('Time step', fontsize=12)
                ax1.set_ylabel('Illicit F1', fontsize=12)
                ax1.set_xticks(range(35, 50))
                ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax2.set_ylabel('Num. samples', fontsize=12)
                
                # Add legend
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
                
                # Display the plot in Streamlit
                st.pyplot(fig)
                plt.close(fig)

# Add information about the models
st.sidebar.markdown("---")
st.sidebar.subheader("About the Models")
st.sidebar.markdown("""
- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees
- **XGBoost**: An optimized distributed gradient boosting library designed to be highly efficient
""")

# Add information about the metrics
st.sidebar.markdown("---")
st.sidebar.subheader("About the Metrics")
st.sidebar.markdown("""
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Cohen Kappa**: Measure of agreement between predictions and actual values
- **Matthews Correlation**: Measure of the quality of binary classifications
""") 