"""
Utility functions for generating and exporting reports and visualizations
from the Patient Lab Results Visualization application.
"""
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import streamlit as st
from datetime import datetime

def generate_report_html(df_patient, patient_id, patient_name, analyte_plots=None):
    """
    Generate an HTML report for the patient's lab results
    
    Parameters:
    -----------
    df_patient : pandas DataFrame
        DataFrame containing the patient's lab results
    patient_id : str
        Patient ID
    patient_name : str
        Patient name
    analyte_plots : dict, optional
        Dictionary of matplotlib figures for each analyte
        
    Returns:
    --------
    html : str
        HTML report
    """
    # Start building the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lab Results Report - Patient {patient_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-bottom: 1px solid #dee2e6; }}
            .section {{ margin-top: 30px; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .abnormal {{ background-color: #ffcccc; }}
            .plot-container {{ margin: 20px 0; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #6c757d; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Laboratory Results Report</h1>
            <p><strong>Patient ID:</strong> {patient_id}</p>
            <p><strong>Patient Name:</strong> {patient_name}</p>
            <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        
        <div class="section summary">
            <h2>Results Summary</h2>
            <p><strong>Total Tests:</strong> {df_patient['Test Name'].nunique()}</p>
            <p><strong>Total Results:</strong> {len(df_patient)}</p>
            <p><strong>Abnormal Results:</strong> {df_patient['Abnormal Flag'].sum()} 
               ({(df_patient['Abnormal Flag'].sum() / len(df_patient) * 100):.1f}%)</p>
            <p><strong>Date Range:</strong> 
               {df_patient['Collection Date'].min().strftime('%Y-%m-%d')} to 
               {df_patient['Collection Date'].max().strftime('%Y-%m-%d')}</p>
        </div>
    """
    
    # Add table of results
    html += """
        <div class="section">
            <h2>Lab Results</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Test</th>
                    <th>Analyte</th>
                    <th>Result</th>
                    <th>Units</th>
                    <th>Reference Range</th>
                </tr>
    """
    
    # Sort results by date (most recent first) and then by test name
    sorted_results = df_patient.sort_values(['Collection Date', 'Test Name'], 
                                           ascending=[False, True])
    
    # Add rows for each result
    for _, row in sorted_results.iterrows():
        # Add CSS class for abnormal results
        row_class = ' class="abnormal"' if row['Abnormal Flag'] else ''
        
        html += f"""
                <tr{row_class}>
                    <td>{row['Collection Date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['Test Name']}</td>
                    <td>{row['Analyte Name']}</td>
                    <td>{row['Result']}</td>
                    <td>{row['Units']}</td>
                    <td>{row['Reference Range']}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    # Add plots if provided
    if analyte_plots:
        html += """
        <div class="section">
            <h2>Result Trends</h2>
        """
        
        for analyte, plot_fig in analyte_plots.items():
            # Convert plot to base64-encoded image
            buf = io.BytesIO()
            plot_fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
            html += f"""
            <div class="plot-container">
                <h3>{analyte}</h3>
                <img src="data:image/png;base64,{img_str}" alt="{analyte} trend" width="100%">
            </div>
            """
        
        html += """
        </div>
        """
    
    # Close the HTML document
    html += """
        <div class="footer">
            <p>This report was generated automatically. Please consult with a healthcare provider for interpretation.</p>
        </div>
    </body>
    </html>
    """
    
    return html

def get_download_link(html, filename="lab_report.html"):
    """
    Generate a download link for the HTML report
    
    Parameters:
    -----------
    html : str
        HTML content
    filename : str, optional
        Name of the file to download
        
    Returns:
    --------
    href : str
        HTML hyperlink for download
    """
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Report</a>'
    return href

def generate_pdf_report(df_patient, patient_id, patient_name):
    """
    Placeholder for PDF report generation
    
    In a real application, this would use a library like ReportLab or WeasyPrint
    to convert the HTML report to a PDF file.
    """
    # This is a placeholder - in a real application, would generate a PDF
    st.warning("PDF generation would be implemented here. This feature requires additional libraries.")
    return None
