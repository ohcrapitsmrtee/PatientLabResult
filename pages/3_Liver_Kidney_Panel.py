import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

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
                import re
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
        ref_df = pd.read_csv(csv_path)
        ref_df.columns = ref_df.columns.str.strip()
        
        # Parse reference ranges from the CSV
        def parse_range_for_ref(r):
            if isinstance(r, str):
                r_clean = str(r).strip()
                
                # Handle different reference range formats
                if r_clean.startswith('>'):
                    try:
                        value = float(r_clean.replace('>', '').strip())
                        return {'Ref Operator': '>', 'Ref Low': value, 'Ref High': pd.NA}
                    except:
                        pass
                elif r_clean.startswith('<'):
                    try:
                        value = float(r_clean.replace('<', '').strip())
                        return {'Ref Operator': '<', 'Ref Low': pd.NA, 'Ref High': value}
                    except:
                        pass
                elif '-' in r_clean:
                    import re
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
        try:
            merge_columns = ['Test Name', 'Analyte Name']
            if all(col in ref_df.columns for col in merge_columns):
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
                
                if 'Units_ref' in df.columns:
                    df['Units'] = df['Units_ref'].combine_first(df.get('Units', pd.Series(None, index=df.index)))
                if 'Analyte #_ref' in df.columns:
                    df['Analyte #'] = df['Analyte #_ref'].combine_first(df.get('Analyte #', pd.Series(None, index=df.index)))
                    
                to_drop = [col for col in ['Ref Operator_ref', 'Ref Low_ref', 'Ref High_ref', 'Units_ref', 'Analyte #_ref'] if col in df.columns]
                if to_drop:
                    df = df.drop(columns=to_drop)
        except Exception as e:
            print(f"Error during merge: {str(e)}")
    
    # Map abnormal flags to boolean
    df['Abnormal Flag'] = df.get('Abnormal Flag', pd.Series(False, index=df.index))
    
    # Override flag based on merged reference operator and numeric result
    def compute_flag(row):
        if 'Result' not in row:
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
            return False
    df['Abnormal Flag'] = df.apply(compute_flag, axis=1)
    
    return df

# Filter by patient
def filter_patient(df, patient_id):
    try:
        patient_id_int = int(patient_id)
        return df[df['Patient ID'] == patient_id_int].copy()
    except (ValueError, TypeError):
        return df[df['Patient ID'].astype(str) == str(patient_id)].copy()

# Create liver/kidney panel heatmap
def create_liver_kidney_heatmap(df, patient_id):
    # Filter for liver/kidney panel analytes
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
    
    # Filter dataframe for panel analytes only
    panel_df = df[df['Analyte Name'].apply(is_panel_analyte)].copy()
    
    if len(panel_df) < 2:
        st.info("No liver/kidney panel data found for this patient.")
        return
    
    # Get unique dates and analytes
    dates = sorted(panel_df['Collection Date'].dt.date.unique())
    analytes = panel_df['Analyte Name'].unique()
    
    if len(dates) <= 1:
        st.info("Need multiple dates to create a meaningful heatmap.")
        return
    
    if len(analytes) == 0:
        st.info("No liver/kidney panel analytes found in this patient's results.")
        return
    
    # Create a normalized score for each result
    df_pivot = pd.DataFrame(index=dates, columns=analytes)
    
    # Create a z-score for each analyte result
    for analyte in analytes:
        analyte_df = panel_df[panel_df['Analyte Name'] == analyte]
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
    
    # Remove columns that are all zeros (no data)
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
    
    # Create heatmap - if we have enough data
    if not df_pivot.empty and len(df_pivot) > 0 and len(df_pivot.columns) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(df_pivot.columns) * 1.2), max(6, len(dates) * 0.5)))
        
        # Use a diverging colormap - blue for low values, red for high values
        cmap = plt.cm.RdBu_r
        
        # Plot heatmap
        sns.heatmap(df_pivot, cmap=cmap, center=0, 
                   vmin=-3, vmax=3, 
                   linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Standard Deviations from Normal Range'},
                   annot=True, fmt='.1f', annot_kws={'size': 8})
        
        ax.set_title(f"Liver/Kidney Panel - Patient {patient_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Lab Test", fontweight='bold')
        ax.set_ylabel("Collection Date", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add summary statistics
        st.subheader("Panel Summary")
        
        # Count abnormal results by analyte
        abnormal_counts = {}
        total_counts = {}
        
        for analyte in analytes:
            analyte_df = panel_df[panel_df['Analyte Name'] == analyte]
            total_counts[analyte] = len(analyte_df)
            abnormal_counts[analyte] = analyte_df['Abnormal Flag'].sum()
        
        # Create summary table
        summary_data = []
        for analyte in analytes:
            total = total_counts[analyte]
            abnormal = abnormal_counts[analyte]
            pct = (abnormal / total * 100) if total > 0 else 0
            summary_data.append({
                'Lab Test': analyte,
                'Total Results': total,
                'Abnormal Results': abnormal,
                'Abnormal %': f"{pct:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Add interpretation guide
        st.info("""
        **Liver/Kidney Panel Heatmap Guide:**
        - **Dark Red** cells: Values significantly ABOVE normal range
        - **Dark Blue** cells: Values significantly BELOW normal range  
        - **White/Light** cells: Values within normal range
        - **Numbers** in cells show the deviation score
        
        **Key Tests:**
        - **BUN & Creatinine/eGFR**: Kidney function markers
        - **ALT, AST, ALKP**: Liver enzyme markers
        - **Albumin & Total Protein**: Liver synthesis function
        - **Bilirubin**: Liver metabolism marker
        """)
    else:
        st.info("Not enough liver/kidney panel data to create a meaningful heatmap.")

# Main Streamlit app
st.title('🩺 Liver/Kidney Panel Heatmap')
st.write("Focused analysis of key liver and kidney function markers across time.")

# Add navigation info
st.info("""
📌 **Other Views:**
- [Main App](/) - Overall abnormal trends screening
- [Complete Lab Viewer](Complete_Lab_Viewer) - All lab values detailed view
- [Patient Test Heatmap](Patient_Test_Heatmap) - Cross-patient comparisons
""")

# Load data
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

# Create patient selection interface
st.subheader("Patient Selection")

# Debug: Check what columns are available
st.write("Debug - Available columns:", df.columns.tolist())

# Get all patients - handle missing columns gracefully
required_cols = ['Patient ID', 'Last Name', 'First Name']
available_cols = [col for col in required_cols if col in df.columns]

if not available_cols:
    st.error("No patient identification columns found in the data. Required: Patient ID, Last Name, First Name")
    st.stop()

# Use available columns to create patient list
if 'Patient ID' in df.columns:
    if len(available_cols) >= 2:
        patients = df[available_cols].drop_duplicates()
    else:
        # Fallback to just Patient ID
        patients = df[['Patient ID']].drop_duplicates()
        patients['Last Name'] = 'Unknown'
        patients['First Name'] = 'Unknown'
else:
    st.error("Patient ID column not found in the data.")
    st.stop()

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
        if 'Last Name' in patients.columns and 'First Name' in patients.columns:
            patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in filtered_patients.iterrows()]
        else:
            patient_options = [f"{row['Patient ID']}" for _, row in filtered_patients.iterrows()]
    else:
        st.warning(f"No patients found with ID containing '{patient_id_search}'")
        if 'Last Name' in patients.columns and 'First Name' in patients.columns:
            patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in patients.iterrows()]
        else:
            patient_options = [f"{row['Patient ID']}" for _, row in patients.iterrows()]
else:
    # No search term - show all patients
    if 'Last Name' in patients.columns and 'First Name' in patients.columns:
        patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in patients.iterrows()]
    else:
        patient_options = [f"{row['Patient ID']}" for _, row in patients.iterrows()]

# Let user select from available or filtered Patient IDs
selected_patient = st.selectbox('Select Patient', patient_options)
patient_id = selected_patient.split(' - ')[0]  # Extract ID from selection

# Advanced settings
with st.expander("Display Settings"):
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

# Generate heatmap button
if st.button('Generate Liver/Kidney Panel Heatmap') or st.checkbox("Auto-generate on selection", value=True):
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
            if 'Last Name' in patient_info and 'First Name' in patient_info:
                st.write(f"**Name:** {patient_info.get('Last Name', 'Unknown')}, {patient_info.get('First Name', 'Unknown')}")
            else:
                st.write("**Name:** Not available")
        with info_col3:
            if 'D.O.B.' in patient_info and pd.notna(patient_info['D.O.B.']):
                st.write(f"**DOB:** {patient_info['D.O.B.']}")
            elif 'Collection Date' in patient_info and pd.notna(patient_info['Collection Date']):
                st.write(f"**Latest Collection:** {patient_info['Collection Date'].strftime('%Y-%m-%d')}")
            else:
                st.write("**DOB:** Not available")
        
        # Generate the liver/kidney panel heatmap
        create_liver_kidney_heatmap(df_patient, patient_id)
