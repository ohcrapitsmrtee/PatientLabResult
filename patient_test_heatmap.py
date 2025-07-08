"""
Module for creating cross-patient heatmap visualizations
Used in the Patient Lab Result visualization application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def create_patient_test_heatmap(df, patient_ids, threshold=None):
    """
    Create a heatmap with patients on Y-axis and tests on X-axis
    
    Parameters:
    -----------
    df : pandas DataFrame
        The full dataframe containing lab results
    patient_ids : list
        A list of patient IDs to include in the heatmap
    threshold : float, optional
        Abnormal threshold percentage to filter tests, default None (no filter)
    
    Returns:
    --------
    fig : matplotlib Figure
        The heatmap figure
    heatmap_data : pandas DataFrame
        The data used to generate the heatmap
    """
    # Filter the dataframe to only include the selected patients
    df_filtered = df[df['Patient ID'].isin(patient_ids)]
    
    # Get unique tests from the filtered data
    tests = [t for t in df_filtered['Test Name'].unique().tolist() if pd.notna(t)]
    
    # Use the provided patient_ids for the heatmap index
    selected_patients = patient_ids
    
    # Create a pivot table: rows=patients, columns=tests, values=% abnormal
    heatmap_data = pd.DataFrame(index=selected_patients, columns=tests)
    
    # For each patient and test combination, calculate percentage of abnormal results
    for patient_id in selected_patients:
        patient_df = df_filtered[df_filtered['Patient ID'] == patient_id]
        
        for test in tests:
            test_results = patient_df[patient_df['Test Name'] == test]
            if len(test_results) > 0:
                abnormal_flags = test_results['Abnormal Flag'].dropna()
                if len(abnormal_flags) > 0:
                    abnormal_pct = abnormal_flags.mean() * 100
                    heatmap_data.loc[patient_id, test] = abnormal_pct
    
    # Fill NAs with 0 (meaning no abnormal results)
    heatmap_data = heatmap_data.fillna(0).astype(float)
    
    # Filter tests if a threshold is provided
    if threshold is not None:
        tests_to_keep = [test for test in tests if (heatmap_data[test] >= threshold).any()]
        
        if tests_to_keep:
            heatmap_data = heatmap_data[tests_to_keep]
        else:
            heatmap_data = pd.DataFrame(index=selected_patients, columns=['No tests meet threshold']).fillna(0)
            
    # Create heatmap, allocating more space per patient and reducing font size
    fig, ax = plt.subplots(figsize=(12, max(6, len(selected_patients) * 0.6)))
    
    if heatmap_data.empty or heatmap_data.columns.empty:
        ax.text(0.5, 0.5, "No data available for the selected patients.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    else:
        sns.heatmap(heatmap_data, cmap=plt.cm.Reds, vmin=0, vmax=100, 
                   linewidths=0.5, ax=ax, cbar_kws={'label': 'Percentage of Abnormal Results'})
        
        ax.set_title("Abnormal Results by Patient and Test")
        ax.set_xlabel("Test Name")
        ax.set_ylabel("Patient ID")
        plt.xticks(rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=8)  # Adjust y-axis font size
    
    plt.tight_layout()
    
    return fig, heatmap_data

def add_patient_names_to_heatmap(df, patient_ids):
    """
    Create a mapping of patient IDs to patient names for heatmap labeling
    
    Parameters:
    -----------
    df : pandas DataFrame
        The full dataframe containing lab results
    patient_ids : list
        List of patient IDs to map to names
    
    Returns:
    --------
    patient_labels : dict
        Dictionary mapping patient IDs to formatted names
    """
    patient_labels = {}
    
    for patient_id in patient_ids:
        # Skip any NA patient IDs
        if pd.isna(patient_id):
            continue
            
        # Get patient data, safely handling empty results
        patient_data = df[df['Patient ID'] == patient_id]
        if len(patient_data) == 0:
            patient_labels[patient_id] = f"{patient_id} - Unknown"
            continue
            
        # Get the first row of data for this patient
        first_patient_row = patient_data.iloc[0]
        
        # Make sure Last Name and First Name exist and are not NA
        last_name = first_patient_row.get('Last Name', 'Unknown')
        first_name = first_patient_row.get('First Name', 'Unknown')
        
        if pd.isna(last_name):
            last_name = 'Unknown'
        if pd.isna(first_name):
            first_name = 'Unknown'
            
        # Create the label
        patient_labels[patient_id] = f"{patient_id} - {last_name}, {first_name}"
    
    return patient_labels
