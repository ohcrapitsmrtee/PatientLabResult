"""
Page for visualizing patient-test heatmaps in the Lab Results Visualization application
This page shows a comparison heatmap with patients on Y-axis and tests on X-axis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sys
import os

# Add parent directory to path to access shared functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lab_trends import load_data, preprocess_data
from patient_test_heatmap import create_patient_test_heatmap, add_patient_names_to_heatmap

st.title('🔥 Patient vs. Test Heatmap')
st.write("This view shows a heatmap of abnormal test results across multiple patients.")

# Load data
uploaded_file = st.file_uploader('Upload CSV', type='csv', key='heatmap_csv_uploader')
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    try:
        data = load_data('Clinical Lab Results_250518095247540.csv')
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a CSV file.")
        st.stop()

df = preprocess_data(data)

# Settings for the heatmap
st.subheader("Heatmap Settings")

# Get total number of patients
total_patients = df['Patient ID'].nunique()
col1, col2 = st.columns(2)

with col1:
    # Slider for number of patients to display
    num_patients = st.slider("Number of patients to display", 
                           min_value=5, 
                           max_value=min(50, total_patients), 
                           value=min(10, total_patients),
                           step=5)
    
    # Option to display patient names instead of just IDs
    show_names = st.checkbox("Show patient names", value=True)

with col2:
    # Option to filter tests by abnormal percentage threshold
    use_threshold = st.checkbox("Only show tests with abnormal results", value=True)
    
    if use_threshold:
        threshold = st.slider("Minimum abnormal percentage", 
                            min_value=0, 
                            max_value=100, 
                            value=10,
                            step=5)
    else:
        threshold = None

    # Sort options
    sort_by = st.selectbox("Sort patients by:", 
                         ["Patient ID", "Most Abnormal First", "Most Tests First"])

# Create an ordering for patients based on the sort selection
patients = df['Patient ID'].unique().tolist()

if sort_by == "Most Abnormal First":
    # Calculate overall abnormal percentage for each patient
    patient_abnormal_pcts = {}
    for patient_id in patients:
        patient_df = df[df['Patient ID'] == patient_id]
        # Handle NA values by using nanmean to ignore them
        abnormal_values = patient_df['Abnormal Flag'].dropna().values
        if len(abnormal_values) > 0:
            patient_abnormal_pcts[patient_id] = np.mean(abnormal_values) * 100
        else:
            patient_abnormal_pcts[patient_id] = 0
    
    # Sort patients by abnormal percentage (descending)
    patients = sorted(patients, key=lambda x: patient_abnormal_pcts.get(x, 0), reverse=True)
    
elif sort_by == "Most Tests First":
    # Count unique tests for each patient
    patient_test_counts = {}
    for patient_id in patients:
        patient_df = df[df['Patient ID'] == patient_id]
        patient_test_counts[patient_id] = patient_df['Test Name'].nunique()
    
    # Sort patients by test count (descending)
    patients = sorted(patients, key=lambda x: patient_test_counts.get(x, 0), reverse=True)
    
else:  # Default: Sort by Patient ID
    # Convert to string first to handle any NA values safely
    patients = sorted([str(p) for p in patients if pd.notna(p)])
    # Convert back to integers if possible
    patients = [int(p) if p.isdigit() else p for p in patients]

# Select patients to display
display_patients = patients[:num_patients]

# Generate the heatmap
if st.button("Generate Heatmap"):
    with st.spinner("Creating heatmap..."):
        # Filter dataframe to only include the selected patients
        filtered_df = df[df['Patient ID'].isin(display_patients)]
          # Add diagnostic information
        st.write(f"Selected patients: {len(display_patients)}")
        st.write(f"Filtered data shape: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
        
        if filtered_df.empty:
            st.error("No data available for the selected patients. Try selecting different patients or adjusting filters.")
            st.stop()
        
        # Create the heatmap
        fig, heatmap_data = create_patient_test_heatmap(filtered_df, num_patients=num_patients, threshold=threshold)
        
        # Display information about the heatmap data
        st.write(f"Heatmap data shape: {heatmap_data.shape[0]} patients × {heatmap_data.shape[1]} tests")
        
        # Display the heatmap
        st.pyplot(fig)
        
        # Add interpretation info
        st.info("""
        **Heatmap Interpretation:**
        - **Darker red** cells indicate a higher percentage of abnormal results for that patient-test combination
        - **White/Light** cells indicate normal results or no data for that combination
        - The color intensity represents the percentage of results that are flagged as abnormal
        """)
        
        # Option to download the heatmap data
        csv_data = heatmap_data.to_csv(index=True)
        st.download_button(
            label="Download Heatmap Data as CSV",
            data=csv_data,
            file_name="patient_test_heatmap.csv",
            mime="text/csv"
        )
          # Show tabular data
        with st.expander("View Heatmap Data as Table"):
            # Check if heatmap_data is empty
            if heatmap_data.empty:
                st.write("No data available to display.")
            else:
                # Get patient names for more readable display
                if show_names:
                    patient_labels = add_patient_names_to_heatmap(df, display_patients)
                    labeled_data = heatmap_data.copy()
                    
                    # Convert index to same type as patient_labels keys for proper lookup
                    # and handle any type mismatches
                    new_index = []
                    for pid in labeled_data.index:
                        # Convert to string for comparison if needed
                        str_pid = str(pid)
                        # Try to find label by various methods
                        if pid in patient_labels:
                            new_index.append(patient_labels[pid])
                        elif str_pid in patient_labels:
                            new_index.append(patient_labels[str_pid])
                        else:
                            # If no label found, keep the original ID
                            new_index.append(f"{pid}")
                    
                    labeled_data.index = new_index
                    st.dataframe(labeled_data)
                    
                    # Also show data dimensions to help debug
                    st.caption(f"Data dimensions: {labeled_data.shape[0]} patients × {labeled_data.shape[1]} tests")
                else:
                    st.dataframe(heatmap_data)
                    st.caption(f"Data dimensions: {heatmap_data.shape[0]} patients × {heatmap_data.shape[1]} tests")

# Add sidebar with explanation
st.sidebar.title("Navigation")
st.sidebar.info("Use the pages in the sidebar to switch between different visualizations.")
st.sidebar.markdown("---")
st.sidebar.info("""
### How to Use the Heatmap
1. Select the number of patients to display
2. Choose whether to filter tests by abnormal threshold
3. Select sorting options for patients
4. Click "Generate Heatmap" to create the visualization
5. Use the expander to see the data in tabular form
""")
