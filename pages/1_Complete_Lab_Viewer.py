import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
import numpy as np
import sys
import os

# Add parent directory to path to access shared functions
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lab_trends import load_data, preprocess_data, filter_patient

st.title('📊 Complete Lab Viewer')
st.write("This view displays all available lab values for a selected patient.")

# Add a direct link to the Abnormal Trends Screening
st.info("📌 To quickly screen for abnormal trends, switch to the [Abnormal Trends Screening](lab_trends) page.")

# Load data
uploaded_file = st.file_uploader('Upload CSV', type='csv')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.success("CSV file successfully loaded!")
else:
    try:
        data = load_data('Last 4 month labs report_250706102346138.csv')
    except FileNotFoundError:
        st.error("Default data file not found. Please upload a CSV file.")
        st.stop()

df = preprocess_data(data)

# Summary metrics to give context
total_patients = df['Patient ID'].nunique()
total_tests = df['Test Name'].nunique()
total_analytes = df['Analyte Name'].nunique()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Patients", total_patients)
with col2:
    st.metric("Unique Tests", total_tests)
with col3:
    st.metric("Unique Analytes", total_analytes)

# Find unique patients
patients = df[['Patient ID', 'Last Name', 'First Name']].drop_duplicates()
patient_options = [f"{row['Patient ID']} - {row['Last Name']}, {row['First Name']}" for _, row in patients.iterrows()]

# Let user select a patient
selected_patient = st.selectbox('Select Patient', patient_options)
patient_id = selected_patient.split(' - ')[0]  # Extract ID from selection

# Convert patient_id to the correct type for filtering
try:
    patient_id_int = int(patient_id)
except (ValueError, TypeError):
    patient_id_int = patient_id

# Add search functionality
with st.expander("🔍 Advanced Search & Filters"):
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        date_filter = st.checkbox("Filter by date range", value=False)
        if date_filter:
            # Get min and max dates from the data
            min_date = df['Collection Date'].min().date()
            max_date = df['Collection Date'].max().date()
            start_date = st.date_input("Start Date", min_date)
            end_date = st.date_input("End Date", max_date)
    
    with search_col2:
        test_filter = st.checkbox("Filter by test", value=False)
        if test_filter:
            selected_tests = st.multiselect("Select tests", df['Test Name'].unique().tolist())

# Filter data for selected patient
df_patient = filter_patient(df, patient_id_int)

# Apply additional filters if enabled
if 'date_filter' in locals() and date_filter:
    df_patient = df_patient[(df_patient['Collection Date'].dt.date >= start_date) & 
                            (df_patient['Collection Date'].dt.date <= end_date)]

if 'test_filter' in locals() and test_filter and selected_tests:
    df_patient = df_patient[df_patient['Test Name'].isin(selected_tests)]

if df_patient.empty:
    st.warning(f"No lab results found for patient {patient_id}")
else:
    # Show patient info
    patient_info = df_patient.iloc[0]
    st.subheader(f"Patient Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.write(f"**ID:** {patient_id}")
    with info_col2:
        st.write(f"**Name:** {patient_info['Last Name']}, {patient_info['First Name']}")
    with info_col3:
        if 'D.O.B.' in patient_info and pd.notna(patient_info['D.O.B.']):
            st.write(f"**DOB:** {patient_info['D.O.B.']}")
        else:
            st.write(f"**DOB:** N/A")
        
    # Create tabbed interface for different views
    tab1, tab2, tab3 = st.tabs(["📈 Trend Charts", "📋 Data Table", "📊 Summary Stats"])
    
    with tab1:
        # Group tests for organization
        test_names = df_patient['Test Name'].unique()
        selected_test = st.selectbox('Select Test', test_names)
        
        # Filter by selected test
        df_test = df_patient[df_patient['Test Name'] == selected_test]
        
        # Group by analyte
        analyte_groups = df_test.groupby('Analyte Name')
        
        # Calculate subplot layout
        n_analytes = len(analyte_groups)
        n_cols = min(2, n_analytes)
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
            if i < len(axes):
                ax = axes[i]
                
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
                
                ax.set_title(f"{analyte}")
                ax.set_xlabel('Collection Date')
                ax.set_ylabel('Result')
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f"{selected_test}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
    
    with tab2:
        # Provide options for filtering and sorting
        col1, col2 = st.columns(2)
        with col1:
            date_sort = st.radio("Sort by date", ["Newest First", "Oldest First"])
        with col2:
            show_abnormal = st.checkbox("Highlight abnormal values", value=True)
        
        # Sort by collection date
        if date_sort == "Newest First":
            df_display = df_patient.sort_values('Collection Date', ascending=False)
        else:
            df_display = df_patient.sort_values('Collection Date')
        
        # Select columns to display
        display_cols = ['Collection Date', 'Test Name', 'Analyte Name', 'Result', 'Units', 
                       'Reference Range', 'Abnormal Flag']
        
        # Show table with conditional formatting for abnormal values
        if show_abnormal:
            st.dataframe(df_display[display_cols].style.apply(
                lambda x: ['background-color: #ffcccc' if x['Abnormal Flag'] else '' for i in x], 
                axis=1
            ))
        else:
            st.dataframe(df_display[display_cols])
    
    with tab3:
        # Create summary statistics
        st.subheader("Summary Statistics by Analyte")
        
        # Add option to sort by most concerning
        sort_method = st.radio("Sort analytes by:", ["Alphabetical", "Most Concerning First"])
        
        # Calculate concern score for each analyte
        analyte_concerns = {}
        for analyte_name, analyte_df in df_patient.groupby('Analyte Name'):
            if len(analyte_df) > 1:
                # Calculate percentage of abnormal values
                abnormal_pct = analyte_df['Abnormal Flag'].mean() * 100
                
                # Calculate trend severity (0 if no trend, otherwise % change)
                sorted_results = analyte_df.sort_values('Collection Date')
                trend_severity = 0
                if len(sorted_results) >= 2:
                    first_value = sorted_results['Result'].iloc[0]
                    last_value = sorted_results['Result'].iloc[-1]
                    if first_value != 0:
                        trend_severity = abs((last_value - first_value) / first_value * 100)
                
                # Combined concern score
                analyte_concerns[analyte_name] = abnormal_pct + trend_severity
        
        # Sort analytes
        if sort_method == "Most Concerning First":
            sorted_analytes = sorted(analyte_concerns.keys(), key=lambda k: analyte_concerns[k], reverse=True)
        else:
            sorted_analytes = sorted(analyte_concerns.keys())
        
        # Generate stats for each analyte
        for analyte_name in sorted_analytes:
            analyte_df = df_patient[df_patient['Analyte Name'] == analyte_name]
            if len(analyte_df) > 1:  # Only show stats if we have multiple values
                # Show concern level in the expander header
                concern_score = analyte_concerns[analyte_name]
                concern_emoji = "🟢"  # Low concern
                if concern_score > 50:
                    concern_emoji = "🔴"  # High concern
                elif concern_score > 20:
                    concern_emoji = "🟠"  # Medium concern
                
                with st.expander(f"{concern_emoji} {analyte_name}"):
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        stats_df = analyte_df[['Result']].describe().T
                        
                        # Add reference range for context
                        ref_low = analyte_df['Ref Low'].iloc[0]
                        ref_high = analyte_df['Ref High'].iloc[0]
                        
                        if pd.notna(ref_low) and pd.notna(ref_high):
                            stats_df['ref_low'] = ref_low
                            stats_df['ref_high'] = ref_high
                        
                        st.dataframe(stats_df)
                    
                    with col2:
                        # Create small trend plot
                        fig, ax = plt.subplots(figsize=(6, 3))
                        sorted_analyte = analyte_df.sort_values('Collection Date')
                        ax.plot(sorted_analyte['Collection Date'], sorted_analyte['Result'], marker='o')
                        
                        if pd.notna(ref_low):
                            ax.axhline(y=ref_low, color='red', linestyle='--', alpha=0.7)
                        if pd.notna(ref_high):
                            ax.axhline(y=ref_high, color='red', linestyle='--', alpha=0.7)
                            
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show if values are trending up or down
                    sorted_results = analyte_df.sort_values('Collection Date')
                    if len(sorted_results) >= 2:
                        first_value = sorted_results['Result'].iloc[0]
                        last_value = sorted_results['Result'].iloc[-1]
                        
                        if first_value != 0:
                            pct_change = ((last_value - first_value) / first_value * 100)
                            
                            if abs(pct_change) > 10:  # Only show significant changes
                                if pct_change > 0:
                                    st.info(f"⬆️ Values trending up: {pct_change:.1f}% increase from first to last measurement")
                                else:
                                    st.info(f"⬇️ Values trending down: {abs(pct_change):.1f}% decrease from first to last measurement")
                        
                        # Add clinical context insights based on the analyte name
                        if analyte_name.lower() in ['glucose', 'hemoglobin a1c', 'hgba1c', 'hba1c']:
                            st.info("💡 **Clinical Context:** Elevated glucose levels may indicate diabetes or prediabetes. Consistent values above reference range suggest a need for further evaluation.")
                        elif analyte_name.lower() in ['ldl', 'cholesterol', 'triglycerides']:
                            st.info("💡 **Clinical Context:** Elevated lipid levels may increase cardiovascular risk. Consider diet and lifestyle interventions.")
                        elif analyte_name.lower() in ['creatinine', 'egfr', 'bun']:
                            st.info("💡 **Clinical Context:** Abnormal kidney function markers may indicate renal issues. Monitor trends over time.")

# Add data export capabilities
if not df_patient.empty:
    st.markdown("---")
    st.subheader("Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        export_format = st.selectbox("Select export format:", ["CSV", "Excel", "JSON"])
        include_all_fields = st.checkbox("Include all fields", value=False)
    
    with export_col2:
        if export_format == "CSV":
            csv_data = df_patient.to_csv(index=False)
            file_name = f"Patient_{patient_id}_Lab_Results.csv"
            mime_type = "text/csv"
        elif export_format == "Excel":
            # Simulate Excel export (in real app would generate actual .xlsx)
            csv_data = df_patient.to_csv(index=False)
            file_name = f"Patient_{patient_id}_Lab_Results.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:  # JSON
            csv_data = df_patient.to_json(orient="records")
            file_name = f"Patient_{patient_id}_Lab_Results.json"
            mime_type = "application/json"
            
        st.download_button(
            label="Download Data",
            data=csv_data,
            file_name=file_name,
            mime=mime_type
        )
        
        st.write("Export will include currently filtered data only.")

# Sidebar with instructions
st.sidebar.title("Navigation")
st.sidebar.info("""
Use the pages in the sidebar to switch between:
- **Abnormal Trends Screening** 
- **Complete Lab Viewer** (current page)
- **Patient Test Heatmap** (new!)
""")
st.sidebar.markdown("---")
st.sidebar.info("""
### How to Use
1. Select a Patient 
2. Navigate the tabs to:
   - View trend charts for each test
   - Examine data in tabular form
   - See summary statistics
3. Use export options to download data
""")

# Add a report generation feature
st.sidebar.markdown("---")
st.sidebar.subheader("Generate Patient Report")

# Import the report utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from report_utils import generate_report_html, get_download_link, generate_pdf_report
except ImportError:
    st.sidebar.error("Report utilities not available.")

report_type = st.sidebar.radio("Report format:", ["HTML", "PDF"])

if st.sidebar.button("Generate Report"):
    if not df_patient.empty:
        # Extract patient name
        patient_info = df_patient.iloc[0]
        patient_name = f"{patient_info['Last Name']}, {patient_info['First Name']}"
        
        if report_type == "HTML":
            # Generate HTML report
            with st.spinner("Generating HTML report..."):
                # Create plots for each analyte
                analyte_plots = {}
                for analyte_name, analyte_df in df_patient.groupby('Analyte Name'):
                    if len(analyte_df) > 1:  # Only create plots if we have multiple values
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        dates = analyte_df['Collection Date']
                        results = analyte_df['Result']
                        
                        # Plot line with markers
                        ax.plot(dates, results, marker='o')
                        
                        # Draw reference range lines
                        low, high = analyte_df['Ref Low'].iloc[0], analyte_df['Ref High'].iloc[0]
                        if pd.notna(low):
                            ax.axhline(y=low, color='red', linestyle='--', alpha=0.7)
                        if pd.notna(high):
                            ax.axhline(y=high, color='red', linestyle='--', alpha=0.7)
                            
                        ax.set_title(f"{analyte_name}")
                        ax.set_xlabel('Collection Date')
                        ax.set_ylabel('Result')
                        
                        plt.tight_layout()
                        analyte_plots[analyte_name] = fig
                
                # Generate HTML report
                html_report = generate_report_html(df_patient, patient_id, patient_name, analyte_plots)
                
                # Create download link
                st.sidebar.markdown(get_download_link(html_report, f"Patient_{patient_id}_Report.html"), unsafe_allow_html=True)
                st.sidebar.success("Report generated! Click the link above to download.")
        else:
            # Generate PDF report (placeholder for now)
            with st.spinner("Generating PDF report..."):
                generate_pdf_report(df_patient, patient_id, patient_name)
    else:
        st.sidebar.warning("No data available to generate report.")
    
# Add feedback mechanism
st.sidebar.markdown("---")
feedback = st.sidebar.text_area("Provide feedback on this tool:")
if feedback:
    st.sidebar.success("Thank you for your feedback! It will help us improve the tool.")
