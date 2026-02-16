import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers , lower_bound, upper_bound


def flight_price_eda(df, target_col = "Price", result_dir="reports"):
    os.makedirs(result_dir, exist_ok=True)
    # Summary statistics
    summary =[]

    summary.append(df.describe(include='all').T)
    summary.append(f"Dataset Shape: {df.shape}")
    summary.append(f"Missing Values:\n{df.isnull().sum()}")
    summary.append(f"Duplicate Rows: {df.duplicated().sum()}")

    ## save cleaned data

    df_cleaned = df.drop_duplicates()
    df_cleaned.to_csv(os.path.join(result_dir, "cleaned_data.csv"), index=False)


    ## Outlier Detection
    outlier_data = pd.DataFrame()
    if target_col in df.columns:
        outliers, lower, upper = detect_outliers_iqr(df, target_col)
        outlier_data = outliers
        outlier_data.to_csv(os.path.join(result_dir, "outliers.csv"), index=False)  
        summary.append(f"Outliers detected in {target_col}: {outliers.shape[0]}")
        summary.append(f"{target_col} Lower Bound: {lower}, Upper Bound: {upper}")


        ## Boxplot for target variable
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[target_col])
        plt.title(f'Boxplot of {target_col}')
        plt.savefig(os.path.join(result_dir, f'boxplot_{target_col}.png'))
        plt.close()

    ## Correlation Matrix
    num_cols = df.select_dtypes(include=["int64", "float64"])
    if num_cols.shape[1] > 1:
        corr = num_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(result_dir, 'correlation_matrix.png'))
        plt.close()

    ## Save summary report
    with open(os.path.join(result_dir, "summary_report.txt"), "w") as f:
        for item in summary:
            f.write(str(item) + "\n\n")
    return {
        "summary": summary,
        "outliers": outlier_data,
        "cleaned_data": df_cleaned
    }