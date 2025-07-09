import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to path to import functions
sys.path.append('.')
from lab_trends import load_data, preprocess_data

# Load and preprocess the data
print("Loading data...")
df = load_data('Last 4 month labs report_250706102346138.csv')
df = preprocess_data(df)

# Test with a specific patient and analyte
patient_id = 2753
analyte_name = 'AST (SGOT)'

print(f"\nAnalyzing Patient {patient_id}, Analyte: {analyte_name}")

# Filter data
patient_data = df[(df['Patient ID'] == patient_id) & (df['Analyte Name'] == analyte_name)]

if patient_data.empty:
    print("No data found for this patient/analyte combination")
    # Show available analytes for this patient
    available = df[df['Patient ID'] == patient_id]['Analyte Name'].unique()
    print(f"Available analytes: {available[:10]}")  # Show first 10
else:
    print(f"Found {len(patient_data)} records")
    
    # Show the data before and after sorting and deduplication
    print("\nBefore sorting:")
    print(patient_data[['Collection Date', 'Result']].to_string())
    
    sorted_data = patient_data.sort_values('Collection Date')
    print(f"\nAfter sorting (still {len(sorted_data)} records):")
    print(sorted_data[['Collection Date', 'Result']].head(10).to_string())
    
    # Apply deduplication like the fixed plotting function
    deduplicated_data = sorted_data.drop_duplicates(subset=['Collection Date'], keep='first')
    print(f"\nAfter deduplication ({len(deduplicated_data)} records):")
    print(deduplicated_data[['Collection Date', 'Result']].to_string())
    
    # Check if the dates are actually datetime objects
    print(f"\nCollection Date dtype: {deduplicated_data['Collection Date'].dtype}")
    
    # Create a simple plot to see if the issue reproduces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot with duplicates
    ax1.plot(sorted_data['Collection Date'], sorted_data['Result'], marker='o', linewidth=2)
    ax1.set_title(f'With Duplicates - Patient {patient_id} - {analyte_name}')
    ax1.set_xlabel('Collection Date')
    ax1.set_ylabel('Result')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot without duplicates
    ax2.plot(deduplicated_data['Collection Date'], deduplicated_data['Result'], marker='o', linewidth=2)
    ax2.set_title(f'After Deduplication - Patient {patient_id} - {analyte_name}')
    ax2.set_xlabel('Collection Date')
    ax2.set_ylabel('Result')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_plot_comparison.png', dpi=100, bbox_inches='tight')
    print("\nComparison plot saved as debug_plot_comparison.png")
