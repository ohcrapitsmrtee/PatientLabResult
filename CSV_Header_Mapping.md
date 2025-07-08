# CSV Header Mapping Documentation

## Overview
This document explains how the PatientLabResult program maps CSV headers from different source files.

## File Sources

### 1. Clinical Results File
**Primary File:** `Last 4 month labs report_250706071202693.csv`
**Fallback File:** `Clinical Lab Results_250518095247540.csv`

**Headers Used:**
- `Last Name` - Patient's last name
- `First Name` - Patient's first name  
- `Patient ID` - Unique patient identifier
- `Test Name` - Name of the lab test performed
- `Analyte Name` - Specific analyte/component measured
- `Reference Range` - Normal range for the analyte
- `Result` - Actual test result value
- `Units` - Units of measurement
- `Abnormal Flag` - Indicates if result is abnormal
- `Collection Date` - When sample was collected (preferred)
- `Reported Date` - When result was reported (fallback if Collection Date not available)

### 2. LabCorp Reference File
**File:** `Labcorp Reference CSV.csv`

**Headers Used:**
- `Test #` - Test identifier number
- `Test Name` - Name of the lab test (used for merging)
- `Analyte Name` - Specific analyte/component (used for merging)
- `Analyte #` - Analyte identifier number
- `Reference Range` - Standard reference range (parsed into Ref Low/High)
- `Units` - Units of measurement (merged if missing in main data)

## Data Processing Flow

1. **Load Clinical Results**: Program loads the clinical results CSV with patient data
2. **Parse Reference Ranges**: Extracts numeric bounds from "Reference Range" strings
3. **Load Reference Data**: Loads LabCorp reference CSV for standardized ranges
4. **Merge Data**: Combines clinical results with reference data using Test Name + Analyte Name
5. **Compute Flags**: Determines abnormal status based on reference ranges
6. **Generate Visualizations**: Creates trend plots and heatmaps

## Key Mappings

### Patient Information
- **Name**: `Last Name`, `First Name` from clinical results
- **ID**: `Patient ID` from clinical results
- **Date**: `Collection Date` (preferred) or `Reported Date` from clinical results

### Test Information
- **Test**: `Test Name` from clinical results (matched with reference data)
- **Analyte**: `Analyte Name` from clinical results (matched with reference data)
- **Result**: `Result` from clinical results
- **Units**: `Units` from clinical results (supplemented by reference data)
- **Range**: `Reference Range` from clinical results (overridden by reference data)
- **Flag**: `Abnormal Flag` from clinical results (recomputed based on reference ranges)

### Reference Data Override
The program prioritizes reference ranges from the LabCorp Reference CSV over the ranges in the clinical results file. This ensures standardized, accurate reference ranges are used for all calculations.

## File Priority
1. User uploaded file (highest priority)
2. `Last 4 month labs report_250706071202693.csv` (newer data)
3. `Clinical Lab Results_250518095247540.csv` (fallback)
