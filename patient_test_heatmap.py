"""
Module for creating cross-patient heatmap visualizations
Used in the Patient Lab Result visualization application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def create_patient_test_heatmap(df, num_patients=10, threshold=None):
    """
    Create a heatmap with patients on Y-axis and tests on X-axis
    
    Parameters:
    -----------
    df : pandas DataFrame
        The full dataframe containing lab results
    num_patients : int, optional
        Number of patients to include, default 10
    threshold : float, optional
        Abnormal threshold percentage to filter tests, default None (no filter)
    
    Returns:
    --------
    fig : matplotlib Figure
        The heatmap figure
    """
    # Get unique patients and tests, filtering out NA values
    patients = [p for p in df['Patient ID'].unique().tolist() if pd.notna(p)]
    tests = [t for t in df['Test Name'].unique().tolist() if pd.notna(t)]
    
    # Default to showing all patients if there are fewer than the limit
    if len(patients) <= num_patients:
        selected_patients = patients
    else:
        # Otherwise select the first num_patients
        selected_patients = patients[:num_patients]
    
    # Create a pivot table: rows=patients, columns=tests, values=% abnormal
    heatmap_data = pd.DataFrame(index=selected_patients, columns=tests)
    
    # For each patient and test combination, calculate percentage of abnormal results
    for patient_id in selected_patients:
        # Skip NA patient IDs
        if pd.isna(patient_id):
            continue
            
        patient_df = df[df['Patient ID'] == patient_id]
        
        for test in tests:
            # Skip NA test names
            if pd.isna(test):
                continue
                
            test_results = patient_df[patient_df['Test Name'] == test]
            if len(test_results) > 0:
                # Calculate mean of abnormal flags, safely handling NA values
                abnormal_flags = test_results['Abnormal Flag'].dropna()
                if len(abnormal_flags) > 0:
                    abnormal_pct = abnormal_flags.mean() * 100
                    heatmap_data.loc[patient_id, test] = abnormal_pct
    
    # Fill NAs with 0 (meaning no abnormal results)
    heatmap_data = heatmap_data.fillna(0)
    # Ensure all values are numeric floats for plotting
    heatmap_data = heatmap_data.astype(float)
      # Filter tests if threshold is provided
    if threshold is not None:
        # Find tests that have at least one patient with abnormal percentage >= threshold
        tests_to_keep = []
        for test in tests:
            # Skip tests that might have been filtered out
            if test not in heatmap_data.columns:
                continue
                
            if (heatmap_data[test] >= threshold).any():
                tests_to_keep.append(test)
        
        # Filter to only keep tests meeting the threshold
        if tests_to_keep:
            heatmap_data = heatmap_data[tests_to_keep]
        else:
            # If no tests meet the threshold, add a note column
            heatmap_data = pd.DataFrame(index=selected_patients, columns=['No tests meet threshold'])
            heatmap_data.iloc[:, 0] = 0  # Fill with zeros
      # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(selected_patients) * 0.4)))
    
    # Use a sequential colormap - white to red for abnormal percentages
    cmap = plt.cm.Reds
    
    # Check if we have data to plot
    if heatmap_data.empty or heatmap_data.columns.empty:
        # Create an empty plot with a message
        ax.text(0.5, 0.5, "No data available with current settings", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    else:
        # Plot heatmap
        sns.heatmap(heatmap_data, cmap=cmap, 
                   vmin=0, vmax=100, 
                   linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Percentage of Abnormal Results'})
        
        ax.set_title("Abnormal Results by Patient and Test")
        ax.set_xlabel("Test Name")
        ax.set_ylabel("Patient ID")
        plt.xticks(rotation=45, ha='right')
    
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
