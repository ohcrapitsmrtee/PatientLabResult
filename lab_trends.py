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
    elif 'Reported Date' in df.columns:
        # Use Reported Date if Collection Date is not available
        df['Collection Date'] = pd.to_datetime(df['Reported Date'], errors='coerce')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    return df

# Preprocess: parse reference ranges, normalize flags
def preprocess_data(df):
    # Debug: Print columns and first few rows to see what we're working with
    print("=== CSV HEADER MAPPING ===")
    print("Main CSV columns:", df.columns.tolist())
    print("Expected columns for Clinical Results:")
    print("- Patient Info: Last Name, First Name, Patient ID")
    print("- Test Info: Test Name, Analyte Name, Reference Range, Result, Units, Abnormal Flag")
    print("- Date Info: Collection Date (or Reported Date)")
    print("First few rows:", df.head(2))
    print("===========================")
    
    # Parse numeric bounds from 'Reference Range' if present
    def parse_range(r):
        if isinstance(r, str):
            # Handle different reference range formats
            r_clean = str(r).strip()
            
            # Handle ranges with < or > symbols
            if r_clean.startswith('>'):
                # Greater than format: ">59" means lower bound only
                try:
                    value = float(r_clean.replace('>', '').strip())
                    return value, pd.NA  # Lower bound, no upper bound
                except:
                    pass
            elif r_clean.startswith('<'):
                # Less than format: "<5" means upper bound only
                try:
                    value = float(r_clean.replace('<', '').strip())
                    return pd.NA, value  # No lower bound, upper bound
                except:
                    pass
            elif '-' in r_clean:
                # Standard range format: "70-99"
                parts = re.split(r"\s*-\s*", r_clean)
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

    # Convert Result to numeric early, removing non-numeric characters
    # This must happen BEFORE compute_flag function to avoid type comparison errors
    if 'Result' in df.columns:
        # Handle various non-numeric prefixes and suffixes that might appear in lab results
        df['Result'] = df['Result'].astype(str).str.replace('<', '', regex=False)
        df['Result'] = df['Result'].str.replace('>', '', regex=False)
        df['Result'] = df['Result'].str.replace('≤', '', regex=False)  # Less than or equal
        df['Result'] = df['Result'].str.replace('≥', '', regex=False)  # Greater than or equal
        df['Result'] = df['Result'].str.strip()  # Remove any leading/trailing whitespace
        df['Result'] = pd.to_numeric(df['Result'], errors='coerce')

    # Override reference ranges using LabCorp reference CSV if available
    import os
    csv_path = 'Labcorp Reference CSV.csv'
    if os.path.exists(csv_path):
        # Print debug information
        print(f"Loading reference data from {csv_path}")
        ref_df = pd.read_csv(csv_path)
        print(f"Reference CSV columns: {ref_df.columns.tolist()}")
        print("Expected columns for LabCorp Reference:")
        print("- Test #, Test Name, Analyte Name, Analyte #, Reference Range, Units")
        print("- Using: Test Name, Analyte Name (for merging), Analyte #, Reference Range (parsed), Units")
        
        # Strip whitespace from reference CSV column names too
        ref_df.columns = ref_df.columns.str.strip()
        
        # Parse reference ranges from the CSV
        def parse_range_for_ref(r):
            if isinstance(r, str):
                r_clean = str(r).strip()
                
                # Handle different reference range formats
                if r_clean.startswith('>'):
                    # Greater than format: ">59" means lower bound only
                    try:
                        value = float(r_clean.replace('>', '').strip())
                        return {'Ref Operator': '>', 'Ref Low': value, 'Ref High': pd.NA}
                    except:
                        pass
                elif r_clean.startswith('<'):
                    # Less than format: "<5" means upper bound only
                    try:
                        value = float(r_clean.replace('<', '').strip())
                        return {'Ref Operator': '<', 'Ref Low': pd.NA, 'Ref High': value}
                    except:
                        pass
                elif '-' in r_clean:
                    # Standard range format: "70-99"
                    parts = re.split(r"\s*-\s*", r_clean)
                    try:
                        return {'Ref Operator': 'range', 'Ref Low': float(parts[0]), 'Ref High': float(parts[1])}
                    except:
                        pass
            return {'Ref Operator': None, 'Ref Low': pd.NA, 'Ref High': pd.NA}
        
        # Extract reference ranges
        ref_ranges = ref_df['Reference Range'].apply(parse_range_for_ref)
        ref_df['Ref Operator'] = ref_ranges.apply(lambda x: x['Ref Operator'])
        ref_df['Ref Low'] = ref_ranges.apply(lambda x: x['Ref Low'])
        ref_df['Ref High'] = ref_ranges.apply(lambda x: x['Ref High'])
        
        # Merge and override parsed ranges and operator where available
        print("Preparing to merge reference data")
        try:
            # Ensure column names match between dataframes
            merge_columns = ['Test Name', 'Analyte Name']
            for col in merge_columns:
                if col not in df.columns:
                    print(f"Warning: '{col}' not in main dataframe columns: {df.columns.tolist()}")
                if col not in ref_df.columns:
                    print(f"Warning: '{col}' not in reference dataframe columns: {ref_df.columns.tolist()}")
            
            # Merge on the columns that both dataframes have
            if all(col in ref_df.columns for col in merge_columns):
                # Include Analyte # in the merge to preserve the analyte identifier
                columns_to_merge = ['Test Name', 'Analyte Name', 'Ref Operator', 'Ref Low', 'Ref High', 'Units']
                if 'Analyte #' in ref_df.columns:
                    columns_to_merge.append('Analyte #')
                
                df = df.merge(
                    ref_df[columns_to_merge],
                    on=merge_columns, how='left', suffixes=('', '_ref')
                )
                # Combine reference values taking priority from reference CSV
                df['Ref Operator'] = df['Ref Operator_ref'].combine_first(df.get('Ref Operator', pd.Series(None, index=df.index)))
                df['Ref Low'] = df['Ref Low_ref'].combine_first(df.get('Ref Low', pd.Series(pd.NA, index=df.index)))
                df['Ref High'] = df['Ref High_ref'].combine_first(df.get('Ref High', pd.Series(pd.NA, index=df.index)))
                # Also merge Units from reference if not present in main data
                if 'Units_ref' in df.columns:
                    df['Units'] = df['Units_ref'].combine_first(df.get('Units', pd.Series(None, index=df.index)))
                # Preserve Analyte # if it was merged
                if 'Analyte #_ref' in df.columns:
                    df['Analyte #'] = df['Analyte #_ref'].combine_first(df.get('Analyte #', pd.Series(None, index=df.index)))
                    
                to_drop = [col for col in ['Ref Operator_ref', 'Ref Low_ref', 'Ref High_ref', 'Units_ref', 'Analyte #_ref'] if col in df.columns]
                if to_drop:
                    df = df.drop(columns=to_drop)
            else:
                print("Cannot merge - missing required columns in reference data")
        except Exception as e:
            print(f"Error during merge: {str(e)}")
            # Continue without reference merge if there's an error
    
    # Map abnormal flags to boolean
    df['Abnormal Flag'] = df.get('Abnormal Flag', pd.Series(False, index=df.index))
    # Override flag based on merged reference operator and numeric result
    def compute_flag(row):
        # Check if 'Result' column exists in the row
        if 'Result' not in row:
            print("Warning: 'Result' column not found in row")
            # Return the original flag or False if no flag exists
            return row.get('Abnormal Flag', False)
        
        try:
            res = row['Result']
            op = row.get('Ref Operator')
            low = row.get('Ref Low')
            high = row.get('Ref High')
            
            # Ensure result is numeric and not NaN
            if pd.isna(res) or not isinstance(res, (int, float)):
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
            return bool(row.get('Abnormal Flag', False))
        except Exception as e:
            print(f"Error in compute_flag: {str(e)}")
            print(f"Result type: {type(row['Result'])}, Value: {row['Result']}")
            return False
    df['Abnormal Flag'] = df.apply(compute_flag, axis=1)
    
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
            grp = grp.sort_values('Collection Date')  # Sort by date before plotting
            # Remove duplicates by keeping the first occurrence for each date
            grp = grp.drop_duplicates(subset=['Collection Date'], keep='first')
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
                    
                    grp = grp.sort_values('Collection Date')  # Sort by date before plotting
                    # Remove duplicates by keeping the first occurrence for each date
                    grp = grp.drop_duplicates(subset=['Collection Date'], keep='first')
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

    # --- Custom Panel Selection ---
    panel_analytes = [
        "BUN", "creatinine", "eGFR", "ALT", "AST", "ALKP", "albumin", "total protein", "bilirubin"
    ]
    panel_label = "Show only Liver/Kidney Panel (BUN, creatinine/eGFR, ALT, AST, ALKP, albumin, total protein, bilirubin)"
    show_panel = st.checkbox(panel_label, value=True)

    # Get unique analytes (case-insensitive)
    analytes_all = df['Analyte Name'].unique()
    if show_panel:
        # Filter analytes for the panel (case-insensitive, partial match for creatinine/eGFR/bilirubin)
        def is_panel_analyte(analyte):
            a = analyte.lower()
            return (
                "bun" in a or
                "creatinine" in a or
                "egfr" in a or
                "alt" in a or
                "ast" in a or
                "alkp" in a or "alkaline phosphatase" in a or
                "albumin" in a or
                "total protein" in a or
                "bilirubin" in a
            )
        analytes = [a for a in analytes_all if is_panel_analyte(a)]
        if not analytes:
            st.info("No panel analytes found in this patient's results.")
            return
    else:
        analytes = analytes_all

    # Pivot data to create matrix: rows=dates, columns=analytes, values=normalized result values
    dates = sorted(df['Collection Date'].dt.date.unique())

    if len(dates) <= 1 or len(analytes) <= 1:
        st.info("Not enough data to create a meaningful heatmap.")
        return

    # Create a normalized score for each result
    df_pivot = pd.DataFrame(index=dates, columns=analytes)

    # Create a z-score for each analyte result
    for analyte in analytes:
        analyte_df = df[df['Analyte Name'] == analyte]
        # Remove duplicates by keeping the first occurrence for each date
        analyte_df = analyte_df.drop_duplicates(subset=['Collection Date'], keep='first')
        
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
    parser.add_argument('--csv', '-c', default='Last 4 month labs report_250706102346138.csv', help='Path to CSV file')
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
        data = load_data('Last 4 month labs report_250706102346138.csv')
    except FileNotFoundError:
        st.error("Default data file 'Last 4 month labs report_250706102346138.csv' not found. Please upload a CSV file.")
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
            # Check for D.O.B. or other date fields
            if 'D.O.B.' in patient_info and pd.notna(patient_info['D.O.B.']):
                st.write(f"**DOB:** {patient_info['D.O.B.']}")
            elif 'Collection Date' in patient_info and pd.notna(patient_info['Collection Date']):
                st.write(f"**Latest Collection:** {patient_info['Collection Date'].strftime('%Y-%m-%d')}")
            else:
                st.write("**DOB:** Not available")
        
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
            colors = ['#ff9999' if x > 50 else '#ffcc99' if x > 25 else '#99ff99' for x in values]
            
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
            abnormal_pct = (abnormal_count / total_results) * 100 if total_results > 0 else 0