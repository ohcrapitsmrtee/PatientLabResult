import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Lab Trends Viewer",
    page_icon="🧪",
)

# Load and parse CSV
def load_data(path):
    # Read CSV, using the Python engine to handle rows with mismatched columns and skip bad lines
    df = pd.read_csv(
        path,
        dtype={'Patient ID': 'Int64'},
        engine='python',
        on_bad_lines='skip'
    )
    # Parse 'Collection Date' if present
    if 'Collection Date' in df.columns:
        df['Collection Date'] = pd.to_datetime(df['Collection Date'], errors='coerce')
    return df

# Preprocess: parse reference ranges, normalize flags
def preprocess_data(df):
    # Parse numeric bounds from 'Reference Range' if present
    def parse_range(r):
        if isinstance(r, str) and '-' in r:
            parts = re.split(r"\s*-\s*", r)
            try:
                return float(parts[0]), float(parts[1])
            except:
                pass
        return pd.NA, pd.NA

    if 'Reference Range' in df.columns:
        ranges = df['Reference Range'].apply(parse_range)
        df[['Ref Low', 'Ref High']] = pd.DataFrame(ranges.tolist(), index=df.index)
    else:
        df['Ref Low'] = pd.NA
        df['Ref High'] = pd.NA
    # Ensure 'Abnormal Flag' column exists even if CSV parsing skipped it
    if 'Abnormal Flag' not in df.columns:
        df['Abnormal Flag'] = False

    # Override reference ranges using LabCorp cleaned JSON if available
    import os
    json_path = 'labcorp_reference_cleaned.json'
    if os.path.exists(json_path):
        ref_df = pd.read_json(json_path)
        # Rename to match lab_trends columns, including operator
        ref_df = ref_df.rename(columns={
            'test_name': 'Test Name',
            'analyte_name': 'Analyte Name',
            'Ref_operator': 'Ref Operator',
            'Ref_low': 'Ref Low',
            'Ref_high': 'Ref High'
        })
        # Merge and override parsed ranges and operator where available
        df = df.merge(
            ref_df[['Test Name', 'Analyte Name', 'Ref Operator', 'Ref Low', 'Ref High']],
            on=['Test Name', 'Analyte Name'], how='left', suffixes=('', '_ref')
        )
        # Combine JSON values taking priority
        df['Ref Operator'] = df['Ref Operator_ref'].combine_first(df.get('Ref Operator'))
        df['Ref Low'] = df['Ref Low_ref'].combine_first(df['Ref Low'])
        df['Ref High'] = df['Ref High_ref'].combine_first(df['Ref High'])
        df = df.drop(columns=['Ref Operator_ref', 'Ref Low_ref', 'Ref High_ref'])
    
    # Map abnormal flags to boolean
    df['Abnormal Flag'] = df.get('Abnormal Flag', pd.Series(False, index=df.index))
    # Override flag based on merged reference operator and numeric result
    def compute_flag(row):
        res = row['Result']
        op = row.get('Ref Operator')
        low = row.get('Ref Low')
        high = row.get('Ref High')
        if pd.isna(res):
            return False
        # Range operator: outside low-high
        if op == 'range':
            if pd.notna(low) and res < low:
                return True
            if pd.notna(high) and res > high:
                return True
            return False
        # Greater-than operator: abnormal if below low bound
        if op == '>':
            return pd.notna(low) and res < low
        # Less-than operator: abnormal if above high bound
        if op == '<':
            return pd.notna(high) and res > high
        # Default: use original flag (already in boolean form)
        return bool(row['Abnormal Flag'])
    df['Abnormal Flag'] = df.apply(compute_flag, axis=1)
    
    # Convert Result to numeric, removing non-numeric characters
    df['Result'] = pd.to_numeric(df['Result'].astype(str).str.replace('<', '').str.replace('>', ''), errors='coerce')
    
    return df

# Filter by patient
def filter_patient(df, patient_id):
    # Try to convert patient_id to int for comparison with Int64 Patient ID column
    try:
        patient_id_int = int(patient_id)
        return df[df['Patient ID'] == patient_id_int].copy()
    except (ValueError, TypeError):
        # Fallback to string comparison if conversion fails
        return df[df['Patient ID'].astype(str) == str(patient_id)].copy()

# Determine persistent drift
def flag_persistent_drift(grp, threshold=0.5):
    out_of_range = grp.apply(lambda row: 
        (pd.notna(row['Ref Low']) and row['Result'] < row['Ref Low']) or
        (pd.notna(row['Ref High']) and row['Result'] > row['Ref High']), axis=1)
    return out_of_range.mean() >= threshold

# Plot trends with focus on abnormal values
def plot_trends(df, patient_id, drift_threshold=0.5):
    df = df.sort_values('Collection Date')
    
    # Create a heatmap visualization first
    create_heatmap(df)
    
    # Group first by Test Name, then by Analyte Name
    test_groups = df.groupby('Test Name')
    
    # Find analytes with persistent drift to show first
    drift_analytes = []
    for test_name, test_df in test_groups:
        for analyte, grp in test_df.groupby('Analyte Name'):
            if flag_persistent_drift(grp, threshold=drift_threshold):
                drift_analytes.append((test_name, analyte, grp))
    
    # Display warning if drift analytes exist
    if drift_analytes:
        st.warning(f"⚠️ Found {len(drift_analytes)} analytes with persistent drift (>= {drift_threshold*100:.0f}% out of range)")
    
    # Show drift analytes first
    if drift_analytes:
        st.subheader("⚠️ Analytes with Persistent Drift")
        for test_name, analyte, grp in drift_analytes:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            dates = grp['Collection Date']
            results = grp['Result']
            
            # Plot line with markers
            ax.plot(dates, results, marker='o')
            
            # Highlight abnormal points
            abn = grp[grp['Abnormal Flag']]
            if not abn.empty:
                ax.scatter(abn['Collection Date'], abn['Result'], edgecolors='red', facecolors='none', s=100)
            
            # Draw reference range lines
            low, high = grp['Ref Low'].iloc[0], grp['Ref High'].iloc[0]
            if pd.notna(low):
                ax.axhline(y=low, color='red', linestyle='--', alpha=0.7, label='Lower Limit')
            if pd.notna(high):
                ax.axhline(y=high, color='red', linestyle='--', alpha=0.7, label='Upper Limit')
            
            # Add persistent drift marker
            ax.text(dates.iloc[-1], results.iloc[-1], '*', color='red', fontsize=24)
            
            ax.set_title(f"{test_name}: {analyte}")
            ax.set_xlabel('Collection Date')
            ax.set_ylabel('Result')
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            fig.tight_layout()
            st.pyplot(fig)
      # Create a figure for each test
    if st.checkbox("Show all test results", value=False):
        for test_name, test_df in test_groups:
            st.subheader(f"Test: {test_name}")
            
            # Group by analyte within each test
            analyte_groups = test_df.groupby('Analyte Name')
            
            # Calculate number of rows and columns for subplots
            n_analytes = len(analyte_groups)
            n_cols = min(2, n_analytes)  # Maximum 2 columns
            n_rows = (n_analytes + n_cols - 1) // n_cols
            
            # Create figure with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            
            # Handle case of single plot
            if n_analytes == 1:
                axes = np.array([axes])
            
            # Flatten axes array for easy indexing
            if n_analytes > 1:
                axes = axes.flatten()
            
            # Plot each analyte in its own subplot
            for i, (analyte, grp) in enumerate(analyte_groups):
                if i < len(axes):  # Safety check
                    ax = axes[i]
                    
                    dates = grp['Collection Date']
                    results = grp['Result']
                    
                    # Plot line with markers
                    ax.plot(dates, results, marker='o')
                    
                    # Highlight abnormal points
                    abn = grp[grp['Abnormal Flag']]
                    if not abn.empty:
                        ax.scatter(abn['Collection Date'], abn['Result'], edgecolors='red', facecolors='none', s=100)
                    
                    # Draw reference range lines instead of a band
                    low, high = grp['Ref Low'].iloc[0], grp['Ref High'].iloc[0]
                    if pd.notna(low):
                        ax.axhline(y=low, color='red', linestyle='--', alpha=0.7, label='Lower Limit')
                    if pd.notna(high):
                        ax.axhline(y=high, color='red', linestyle='--', alpha=0.7, label='Upper Limit')
                    
                    # Annotate persistent drift
                    if flag_persistent_drift(grp, threshold=drift_threshold):
                        ax.text(dates.iloc[-1], results.iloc[-1], '*', color='red', fontsize=14)
                    
                    ax.set_title(f"{analyte}")
                    ax.set_xlabel('Collection Date')
                    ax.set_ylabel('Result')
                    
                    # Format x-axis dates
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            
            fig.suptitle(f"Patient {patient_id}: {test_name}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            st.pyplot(fig)
    
    return None  # No longer returning a single figure

# Create a heatmap visualization of abnormal results
def create_heatmap(df):
    # Only proceed if we have enough data
    if len(df) < 2:
        return
    
    st.subheader("Lab Results Heatmap")
    st.write("This heatmap shows patterns of abnormal results across tests over time.")
    
    # Pivot data to create matrix: rows=dates, columns=analytes, values=normalized result values
    # First get unique dates and analytes
    dates = sorted(df['Collection Date'].dt.date.unique())
    analytes = df['Analyte Name'].unique()
    
    if len(dates) <= 1 or len(analytes) <= 1:
        st.info("Not enough data to create a meaningful heatmap.")
        return
    
    # Create a normalized score for each result
    df_pivot = pd.DataFrame(index=dates, columns=analytes)
    
    # Create a z-score for each analyte result
    for analyte in analytes:
        analyte_df = df[df['Analyte Name'] == analyte]
        for _, row in analyte_df.iterrows():
            date = row['Collection Date'].date()
            result = row['Result']
            ref_low = row['Ref Low']
            ref_high = row['Ref High']
            
            # Skip rows with missing data
            if pd.isna(result):
                continue
                
            # Calculate how far the result is outside the reference range
            if pd.notna(ref_low) and pd.notna(ref_high):
                # Range is available
                mid_range = (ref_high + ref_low) / 2
                range_width = ref_high - ref_low
                # Normalize to a -3 to +3 scale (like z-scores)
                if range_width > 0:
                    z_score = 2 * (result - mid_range) / range_width
                    df_pivot.loc[date, analyte] = z_score
            else:
                # No range info - just record if flagged as abnormal
                df_pivot.loc[date, analyte] = 1 if row['Abnormal Flag'] else 0
    
    # Fill NAs with 0 (meaning normal)
    df_pivot = df_pivot.fillna(0)
    
    # Create heatmap - if we have enough data
    if not df_pivot.empty and len(df_pivot) > 0 and len(df_pivot.columns) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(dates) * 0.4)))
        
        # Use a diverging colormap - blue for low values, red for high values
        cmap = plt.cm.RdBu_r
        
        # Plot heatmap
        sns.heatmap(df_pivot, cmap=cmap, center=0, 
                   vmin=-3, vmax=3, 
                   linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Standard Deviations from Normal Range'})
        
        ax.set_title(f"Lab Result Patterns for Patient {patient_id}")
        ax.set_xlabel("Analyte")
        ax.set_ylabel("Collection Date")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add interpretation guide
        st.info("""
        **Heatmap Interpretation Guide:**
        - **Dark Red** cells represent values significantly ABOVE normal range
        - **Dark Blue** cells represent values significantly BELOW normal range
        - **White/Light** cells represent values within normal range
        
        Look for patterns across time (rows) or across different lab tests (columns).
        """)
    else:
        st.info("Not enough data to create a meaningful heatmap.")

# Main function with CLI args
def main():
    parser = argparse.ArgumentParser(description='Plot lab analyte trends for a patient')
    parser.add_argument('--patient', '-p', required=True, help='Patient ID to filter')
    parser.add_argument('--csv', '-c', default='Clinical Lab Results_250518095247540.csv', help='Path to CSV file')
    parser.add_argument('--drift-threshold', '-t', type=float, default=0.5, help='Threshold fraction for persistent drift flag')
    args = parser.parse_args()

    df = load_data(args.csv)
    df = preprocess_data(df)
    df_patient = filter_patient(df, args.patient)
    if df_patient.empty:
        print(f'No records found for patient {args.patient}')
        return
    plot_trends(df_patient, args.patient, args.drift_threshold)

# Main Streamlit app - Abnormal Trends Screening
st.title('🚨 Abnormal Lab Trends Screening')
st.write("This view helps identify patients with lab values that are consistently outside the normal range.")

# Add links to other pages
st.info("""
📌 **Available Views:**
- For detailed analysis of all lab values, switch to the [Complete Lab Viewer](Complete_Lab_Viewer) page.
- For cross-patient test comparison, try the [Patient Test Heatmap](Patient_Test_Heatmap) page.
""")

uploaded_file = st.file_uploader('Upload CSV', type='csv')
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    try:
        data = load_data('Clinical Lab Results_250518095247540.csv')
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a CSV file.")
        st.stop()

df = preprocess_data(data)

# Create a simplified search interface for patients - ID only
st.subheader("Patient Selection")

# Get all patients
patients = df[['Patient ID', 'Last Name', 'First Name']].drop_duplicates()

# Create a search box for patient ID
search_col1, search_col2 = st.columns([1, 3])

with search_col1:
    st.write("Search by ID:")
    
with search_col2:
    patient_id_search = st.text_input("Enter Patient ID", key="patient_id_search")

# Filter patients based on ID search
if patient_id_search:
    try:
        # Try exact match with integer conversion
        search_int = int(patient_id_search)
        id_exact_match = patients['Patient ID'] == search_int
        
        # Also try contains match with string conversion
        id_contains_match = patients['Patient ID'].astype(str).str.contains(str(patient_id_search), case=False)
        
        # Combine both match types
        filtered_patients = patients[id_exact_match | id_contains_match]
    except (ValueError, TypeError):
        # If conversion fails, just do string contains match
        filtered_patients = patients[patients['Patient ID'].astype(str).str.contains(str(patient_id_search), case=False)]
    
    if len(filtered_patients) > 0:
        patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in filtered_patients.iterrows()]
    else:
        st.warning(f"No patients found with ID containing '{patient_id_search}'")
        patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in patients.iterrows()]
else:
    # No search term - show all patients
    patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in patients.iterrows()]

# Let user select from available or filtered Patient IDs
selected_patient = st.selectbox('Select Patient', patient_options)
patient_id = selected_patient.split(' - ')[0]  # Extract ID from selection

# Advanced settings in an expander
with st.expander("Advanced Settings"):
    # Add explanation for drift threshold
    st.write("""
    **Drift Threshold:** Fraction of values that must be outside normal range to be flagged as persistently drifting (marked with *).
    A value of 0.5 means that at least half of the results must be outside the reference range.
    """)
    drift_threshold = st.slider('Drift Threshold', 0.0, 1.0, 0.5)
    
    # Add option to filter by date range
    st.write("**Date Range Filter:**")
    date_filter = st.checkbox("Filter by date range", value=False)
    
    if date_filter:
        # Get min and max dates from the data
        min_date = df['Collection Date'].min().date()
        max_date = df['Collection Date'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)

# Add a "Quick Screen" option for immediate results
auto_screen = st.checkbox("Auto-screen on selection", value=False)

if auto_screen or st.button('Screen for Abnormal Trends'):
    df_patient = filter_patient(df, patient_id)
    
    if df_patient.empty:
        st.warning(f"No records found for patient {patient_id}")
    else:
        # Apply date filter if selected
        if 'date_filter' in locals() and date_filter:
            df_patient = df_patient[(df_patient['Collection Date'].dt.date >= start_date) & 
                                    (df_patient['Collection Date'].dt.date <= end_date)]
            
            if df_patient.empty:
                st.warning(f"No records found in selected date range")
                st.stop()
        
        # Show patient info
        patient_info = df_patient.iloc[0]
        st.subheader(f"Patient Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.write(f"**ID:** {patient_id}")
        with info_col2:
            st.write(f"**Name:** {patient_info['Last Name']}, {patient_info['First Name']}")
        with info_col3:
            if 'D.O.B.' in patient_info:
                st.write(f"**DOB:** {patient_info['D.O.B.']}")
        
        # Add a summary visualization of all abnormal results
        st.subheader("Lab Results Overview")
        
        # Calculate abnormal percentages by test
        test_groups = df_patient.groupby('Test Name')
        test_abnormal_pcts = {}
        
        for test_name, test_df in test_groups:
            abnormal_pct = test_df['Abnormal Flag'].mean() * 100
            test_abnormal_pcts[test_name] = abnormal_pct
        
        # Create a bar chart of abnormal percentages
        if test_abnormal_pcts:
            fig, ax = plt.subplots(figsize=(10, 5))
            tests = list(test_abnormal_pcts.keys())
            values = list(test_abnormal_pcts.values())
            
            # Sort by abnormal percentage
            sorted_indices = np.argsort(values)[::-1]
            tests = [tests[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            # Choose colors based on values
            colors = ['#ff9999' if x > 50 else '#ffcc99' if x > 25 else '#99ff999' for x in values]
            
            ax.bar(tests, values, color=colors)
            ax.set_xlabel('Test Name')
            ax.set_ylabel('Abnormal Results (%)')
            ax.set_title('Percentage of Abnormal Results by Test')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Count of abnormal results
            total_results = len(df_patient)
            abnormal_count = df_patient['Abnormal Flag'].sum()
            abnormal_pct = (abnormal_count / total_results) * 100 if total_results > 0