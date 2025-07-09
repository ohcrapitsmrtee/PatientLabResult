import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

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

# Filter for liver/kidney panel analytes
def filter_panel_analytes(df):
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
    
    return df[df['Analyte Name'].apply(is_panel_analyte)].copy()

# Get most recent results for each patient-analyte combination
def get_most_recent_results(df):
    # Sort by collection date descending and get the most recent result for each patient-analyte
    df_sorted = df.sort_values('Collection Date', ascending=False)
    most_recent = df_sorted.groupby(['Patient ID', 'Analyte Name']).first().reset_index()
    return most_recent

# Calculate trend scores (if multiple time points available)
def calculate_trend_scores(df, recent_days=90):
    """Calculate trend scores for each patient-analyte combination"""
    trend_scores = {}
    cutoff_date = datetime.now() - timedelta(days=recent_days)
    
    for (patient_id, analyte), group in df.groupby(['Patient ID', 'Analyte Name']):
        # Filter to recent results only
        recent_results = group[group['Collection Date'] >= cutoff_date].sort_values('Collection Date')
        # Remove duplicates by keeping the first occurrence for each date
        recent_results = recent_results.drop_duplicates(subset=['Collection Date'], keep='first')
        
        if len(recent_results) >= 2:
            # Calculate trend (positive = increasing, negative = decreasing)
            x = np.arange(len(recent_results))
            y = recent_results['Result'].values
            
            # Simple linear trend
            if len(x) > 1 and not np.isnan(y).all():
                slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                
                # Normalize slope by reference range for comparison across tests
                ref_low = recent_results['Ref Low'].iloc[0]
                ref_high = recent_results['Ref High'].iloc[0]
                
                if pd.notna(ref_low) and pd.notna(ref_high):
                    range_width = ref_high - ref_low
                    if range_width > 0:
                        normalized_slope = slope / range_width
                    else:
                        normalized_slope = 0
                else:
                    normalized_slope = 0
                
                trend_scores[(patient_id, analyte)] = normalized_slope
            else:
                trend_scores[(patient_id, analyte)] = 0
        else:
            trend_scores[(patient_id, analyte)] = 0
    
    return trend_scores

# Create population screening heatmap
def create_population_heatmap(df):
    st.subheader("🔍 Population Screening - Liver/Kidney Panel")
    st.write("Bird's-eye view of all patients showing most recent liver/kidney panel results. Red = elevated, needs investigation.")
    
    # Filter for panel analytes
    panel_df = filter_panel_analytes(df)
    
    if len(panel_df) == 0:
        st.info("No liver/kidney panel data found in the dataset.")
        return
    
    # Get most recent results
    most_recent = get_most_recent_results(panel_df)
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        # Patient selection method
        selection_method = st.radio("Patient Selection:", 
                                   ["Auto-select patients", "Manual Patient IDs"],
                                   help="Choose how to select patients for the heatmap")
        
        if selection_method == "Auto-select patients":
            max_patients = st.slider("Maximum patients to show", 10, 100, 50)
            sort_by = st.selectbox("Sort patients by", 
                                  ["Most abnormal results", "Patient ID", "Most recent test date"])
            only_abnormal = st.checkbox("Show only patients with abnormal results", value=False)
        else:
            # Manual patient ID entry
            st.write("**Enter Patient IDs (up to 10):**")
            manual_patient_ids = st.text_area(
                "Patient IDs",
                placeholder="Enter patient IDs separated by commas, spaces, or new lines\nExample: 12345, 67890, 11111",
                help="Enter up to 10 patient IDs. You can separate them with commas, spaces, or put each on a new line.",
                height=100
            )
    
    with col2:
        show_trends = st.checkbox("Include trend indicators", value=True)
        
        if selection_method == "Manual Patient IDs":
            # Parse manual patient IDs
            if manual_patient_ids.strip():
                # Split by various delimiters and clean up
                import re
                patient_ids_raw = re.split(r'[,\s\n]+', manual_patient_ids.strip())
                manual_ids = []
                
                for pid in patient_ids_raw:
                    pid = pid.strip()
                    if pid:
                        try:
                            manual_ids.append(int(pid))
                        except ValueError:
                            st.warning(f"'{pid}' is not a valid patient ID number")
                
                # Limit to 10 patients
                if len(manual_ids) > 10:
                    st.warning(f"Too many patient IDs entered ({len(manual_ids)}). Only the first 10 will be used.")
                    manual_ids = manual_ids[:10]
                
                if manual_ids:
                    st.success(f"Selected {len(manual_ids)} patients: {', '.join(map(str, manual_ids))}")
                else:
                    st.info("No valid patient IDs entered yet.")
            else:
                manual_ids = []
                st.info("Enter patient IDs in the text area to the left.")
    
    # Filter patients based on selection method
    if selection_method == "Manual Patient IDs":
        if not manual_ids:
            st.warning("Please enter at least one valid Patient ID to generate the heatmap.")
            return
        
        # Filter to manually selected patients
        available_patients = most_recent['Patient ID'].unique()
        valid_manual_ids = [pid for pid in manual_ids if pid in available_patients]
        invalid_ids = [pid for pid in manual_ids if pid not in available_patients]
        
        if invalid_ids:
            st.warning(f"Patient IDs not found in data: {', '.join(map(str, invalid_ids))}")
        
        if not valid_manual_ids:
            st.error("None of the entered Patient IDs were found in the dataset.")
            return
        
        top_patients = valid_manual_ids
        st.info(f"Generating heatmap for {len(top_patients)} manually selected patients.")
    
    else:
        # Auto-select patients logic
        # Filter patients if requested
        if only_abnormal:
            abnormal_patients = most_recent[most_recent['Abnormal Flag'] == True]['Patient ID'].unique()
            most_recent_filtered = most_recent[most_recent['Patient ID'].isin(abnormal_patients)]
        else:
            most_recent_filtered = most_recent
        
        # Calculate patient priority scores for sorting
        if sort_by == "Most abnormal results":
            patient_scores = most_recent_filtered.groupby('Patient ID')['Abnormal Flag'].sum().sort_values(ascending=False)
            top_patients = patient_scores.head(max_patients).index.tolist()
        elif sort_by == "Most recent test date":
            patient_dates = most_recent_filtered.groupby('Patient ID')['Collection Date'].max().sort_values(ascending=False)
            top_patients = patient_dates.head(max_patients).index.tolist()
        else:  # Patient ID
            top_patients = sorted(most_recent_filtered['Patient ID'].unique())[:max_patients]
    
    # Filter to top patients
    heatmap_data = most_recent[most_recent['Patient ID'].isin(top_patients)]
    
    if len(heatmap_data) == 0:
        st.warning("No data available for the selected criteria.")
        return
    
    # Calculate trend scores if requested
    trend_scores = {}
    if show_trends:
        trend_scores = calculate_trend_scores(panel_df)
    
    # Create pivot table for heatmap
    # Calculate deviation scores (how far outside normal range)
    def calculate_deviation_score(row):
        result = row['Result']
        ref_low = row['Ref Low']
        ref_high = row['Ref High']
        
        if pd.isna(result):
            return 0
            
        if pd.notna(ref_low) and pd.notna(ref_high):
            # Calculate how many standard deviations outside normal
            mid_range = (ref_high + ref_low) / 2
            range_width = ref_high - ref_low
            
            if range_width > 0:
                deviation = (result - mid_range) / (range_width / 4)  # Rough standard deviation estimate
                return max(-3, min(3, deviation))  # Cap at -3 to +3
        
        # Fallback: use abnormal flag
        return 1 if row['Abnormal Flag'] else 0
    
    heatmap_data['Deviation_Score'] = heatmap_data.apply(calculate_deviation_score, axis=1)
    
    # Create pivot table
    pivot_data = heatmap_data.pivot_table(
        index='Patient ID', 
        columns='Analyte Name', 
        values='Deviation_Score', 
        fill_value=0
    )
    
    # Ensure we have data to plot
    if pivot_data.empty:
        st.warning("No data available to create heatmap.")
        return
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_data.columns) * 0.8), 
                                   max(8, len(pivot_data) * 0.3)))
    
    # Use a colormap emphasizing elevated values (red)
    cmap = plt.cm.RdYlBu_r  # Red for high, blue for low
    
    # Create heatmap with annotations
    sns.heatmap(pivot_data, 
                cmap=cmap, 
                center=0,
                vmin=-2, vmax=3,  # Emphasize elevated values
                linewidths=0.5, 
                ax=ax,
                annot=True, 
                fmt='.1f',
                annot_kws={'size': 8},
                cbar_kws={'label': 'Deviation from Normal (Higher = More Concerning)'})
    
    # Add trend indicators if requested
    if show_trends and trend_scores:
        for i, patient_id in enumerate(pivot_data.index):
            for j, analyte in enumerate(pivot_data.columns):
                trend_score = trend_scores.get((patient_id, analyte), 0)
                if abs(trend_score) > 0.1:  # Significant trend
                    # Add arrow indicator
                    if trend_score > 0:
                        ax.text(j + 0.7, i + 0.3, '↗', fontsize=12, color='darkred', weight='bold')
                    else:
                        ax.text(j + 0.7, i + 0.3, '↘', fontsize=12, color='darkblue', weight='bold')
    
    ax.set_title("Population Screening - Liver/Kidney Panel\n(Most Recent Results)", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Lab Tests", fontweight='bold')
    ax.set_ylabel("Patient ID", fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Create summary statistics
    st.subheader("📊 Summary Statistics")
    
    # Count patients with elevated results by test
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Patients with Elevated Results by Test:**")
        elevated_counts = heatmap_data[heatmap_data['Deviation_Score'] > 1].groupby('Analyte Name').size()
        total_patients_summary = len(heatmap_data['Patient ID'].unique())
        
        if len(elevated_counts) > 0:
            for analyte, count in elevated_counts.items():
                percentage = (count / total_patients_summary) * 100
                st.write(f"• {analyte}: {count}/{total_patients_summary} ({percentage:.1f}%)")
        else:
            st.write("No elevated results found in selected patients.")
    
    with col2:
        if selection_method == "Manual Patient IDs":
            st.write("**Selected Patients Summary:**")
            st.write(f"• Total patients requested: {len(manual_ids)}")
            st.write(f"• Patients found in data: {len(top_patients)}")
            if invalid_ids:
                st.write(f"• Patients not found: {len(invalid_ids)}")
            
            # Show abnormal counts for manual selection
            patient_abnormal_counts = heatmap_data[heatmap_data['Deviation_Score'] > 1].groupby('Patient ID').size()
            if len(patient_abnormal_counts) > 0:
                st.write("**Patients with elevated results:**")
                for patient_id, count in patient_abnormal_counts.sort_values(ascending=False).items():
                    st.write(f"• Patient {patient_id}: {count} elevated tests")
            else:
                st.write("• No elevated results in selected patients")
        else:
            st.write("**High Priority Patients:**")
            # Patients with multiple elevated results
            patient_abnormal_counts = heatmap_data[heatmap_data['Deviation_Score'] > 1].groupby('Patient ID').size()
            high_priority = patient_abnormal_counts[patient_abnormal_counts >= 2].sort_values(ascending=False)
            
            if len(high_priority) > 0:
                for patient_id, count in high_priority.head(10).items():
                    st.write(f"• Patient {patient_id}: {count} elevated tests")
            else:
                st.write("No patients with multiple elevated results.")
    
    # Interpretation guide
    st.info("""
    **Population Screening Guide:**
    
    🔴 **Dark Red**: Significantly elevated - requires immediate attention
    🟡 **Yellow**: Mildly elevated - monitor closely  
    🔵 **Blue**: Below normal - may need investigation
    ⬜ **White**: Normal range
    
    **Trend Indicators** (if enabled):
    ↗ **Rising trend** - values increasing over time
    ↘ **Falling trend** - values decreasing over time
    
    **Priority Focus:**
    - Patients with multiple red cells need immediate review
    - Rising trends in liver enzymes (ALT, AST, ALKP) suggest liver stress
    - Rising BUN/creatinine or falling eGFR suggests kidney issues
    - Low albumin with high bilirubin suggests liver dysfunction
    
    **Manual Selection Benefits:**
    - Focus on specific patients of clinical interest
    - Follow up on previously identified concerns
    - Compare related patients (family members, exposure groups)
    - Track specific cohorts over time
    """)
    
    return heatmap_data

# Main Streamlit app
st.title('🏥 Population Health Screening')
st.write("Identify patients requiring immediate attention based on liver and kidney function markers.")

# Add navigation info
st.info("""
📌 **Other Views:**
- [Main App](/) - Individual patient abnormal trends
- [Complete Lab Viewer](Complete_Lab_Viewer) - Detailed lab analysis
- [Patient Test Heatmap](Patient_Test_Heatmap) - Cross-patient comparisons  
- [Liver/Kidney Panel](Liver_Kidney_Panel) - Individual patient focus
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

# Debug info (expandable)
with st.expander("🔧 Debug Information"):
    st.write("**Available columns:**", df.columns.tolist())
    st.write("**Total records:**", len(df))
    
    # Check for panel analytes
    panel_df = filter_panel_analytes(df)
    st.write("**Panel records found:**", len(panel_df))
    
    if len(panel_df) > 0:
        st.write("**Available panel analytes:**", panel_df['Analyte Name'].unique().tolist())
        st.write("**Date range:**", 
                f"{df['Collection Date'].min().strftime('%Y-%m-%d')} to {df['Collection Date'].max().strftime('%Y-%m-%d')}")

# Create the population heatmap
if st.button('Generate Population Screening Heatmap') or st.checkbox("Auto-generate", value=True):
    create_population_heatmap(df)
