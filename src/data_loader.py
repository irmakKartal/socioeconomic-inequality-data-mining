"""
Data Loader and Preprocessing Module for Socio-Economic Inequality Mining.
Provides utility functions to load, clean, and prepare data for modeling.
"""

import os
import pandas as pd


def get_data_path(filename="inequality_data.csv"):
    """Returns absolute path to a file in the data/ directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", filename)


def load_dataset(filepath=None):
    """
    Loads the cleaned socio-economic inequality dataset.
    Returns:
        pd.DataFrame: Cleaned dataset ready for EDA and ML.
    """
    if filepath is None:
        filepath = get_data_path("inequality_data.csv")

    df = pd.read_csv(filepath)
    return df


def get_numerical_features():
    """Returns the list of analytical numerical columns used in the study."""
    return [
        "gini_coefficient",
        "poverty_rate_pct",
        "income_disparity_ratio",
        "gender_wage_gap_pct",
        "education_quality_index",
        "urban_unemployment_rate",
        "rural_unemployment_rate",
        "unemployment_disparity",
    ]


def get_feature_labels():
    """Human-readable dictionary mapping column names to clean labels."""
    return {
        "gini_coefficient": "Gini Coefficient",
        "poverty_rate_pct": "Poverty Rate (%)",
        "income_disparity_ratio": "Income Disparity Ratio (Top 10% / Bottom 10%)",
        "gender_wage_gap_pct": "Gender Wage Gap (%)",
        "education_quality_index": "Education Quality Index",
        "urban_unemployment_rate": "Urban Unemployment Rate (%)",
        "rural_unemployment_rate": "Rural Unemployment Rate (%)",
        "unemployment_disparity": "Urban-Rural Unemployment Disparity (%)",
    }
