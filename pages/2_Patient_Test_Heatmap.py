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
import re

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
        data = load_data('Last 4 month labs report_250706102346138.csv')
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a CSV file.")
        st.stop()

df = preprocess_data(data)

# Settings for the heatmap
st.subheader("Heatmap Settings")

# Allow user to select patients manually or by sorting
selection_method = st.radio("Patient Selection Method", 
                          ("Select Top Patients by Sorting", "Enter Patient IDs Manually"))

col1, col2 = st.columns(2)

if selection_method == "Enter Patient IDs Manually":
    with col1:
        patient_id_input = st.text_area("Enter up to 10 Patient IDs (separated by commas, spaces, or newlines)", height=150)
        show_names = st.checkbox("Show patient names", value=True)
        
        # Parse patient IDs from input
        if patient_id_input:
            raw_ids = re.split(r'[\s,;\n]+', patient_id_input)
            display_patients = []
            for pid in raw_ids:
                pid = pid.strip()
                if pid:
                    try:
                        display_patients.append(int(pid))
                    except ValueError:
                        st.warning(f"Could not convert '{pid}' to a number; it will be treated as a string.")
                        display_patients.append(pid)
            
            if len(display_patients) > 10:
                st.warning("More than 10 IDs entered. Only the first 10 will be used.")
                display_patients = display_patients[:10]
        else:
            display_patients = []

else:  # Select Top Patients by Sorting
    with col1:
        total_patients = df['Patient ID'].nunique()
        num_patients = st.slider("Number of patients to display", 
                               min_value=5, 
                               max_value=min(50, total_patients), 
                               value=min(10, total_patients),
                               step=5)
        show_names = st.checkbox("Show patient names", value=True)

    with col2:
        sort_by = st.selectbox("Sort patients by:", 
                             ["Patient ID", "Most Abnormal First", "Most Tests First"])

    # Logic for sorting patients
    patients = df['Patient ID'].unique().tolist()
    if sort_by == "Most Abnormal First":
        patient_abnormal_pcts = {pid: df[df['Patient ID'] == pid]['Abnormal Flag'].mean() for pid in patients}
        patients = sorted(patients, key=lambda x: patient_abnormal_pcts.get(x, 0), reverse=True)
    elif sort_by == "Most Tests First":
        patient_test_counts = {pid: df[df['Patient ID'] == pid]['Test Name'].nunique() for pid in patients}
        patients = sorted(patients, key=lambda x: patient_test_counts.get(x, 0), reverse=True)
    else:
        patients = sorted([p for p in patients if pd.notna(p)])
    
    display_patients = patients[:num_patients]

# Heatmap generation settings (common to both methods)
with col2:
    use_threshold = st.checkbox("Only show tests with high abnormal rates", value=True)
    if use_threshold:
        threshold = st.slider("Minimum abnormal percentage for a test to be shown", 
                            min_value=0, max_value=100, value=10, step=5)
    else:
        threshold = None

# Generate the heatmap
if st.button("Generate Heatmap"):
    if not display_patients:
        st.error("Please select or enter at least one patient ID.")
        st.stop()

    with st.spinner("Creating heatmap..."):
        fig, heatmap_data = create_patient_test_heatmap(df, patient_ids=display_patients, threshold=threshold)
        
        if heatmap_data.empty:
            st.warning("No data available for the selected patients or filters.")
            st.stop()

        st.pyplot(fig)
        
        st.info("""
        **Heatmap Interpretation:**
        - **Darker red** cells indicate a higher percentage of abnormal results.
        - **White/Light** cells indicate normal results or no data.
        """)
        
        # Download and data table view
        csv_data = heatmap_data.to_csv(index=True)
        st.download_button("Download Heatmap Data as CSV", csv_data, "patient_test_heatmap.csv", "text/csv")

        with st.expander("View Heatmap Data as Table"):
            if show_names:
                patient_labels = add_patient_names_to_heatmap(df, display_patients)
                labeled_data = heatmap_data.copy()
                labeled_data.index = labeled_data.index.map(patient_labels)
                st.dataframe(labeled_data)
            else:
                st.dataframe(heatmap_data)

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
