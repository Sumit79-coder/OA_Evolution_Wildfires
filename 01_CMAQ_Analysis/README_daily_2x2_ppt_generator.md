# CMAQ Daily 2×2 Fire Impact Maps - PowerPoint Generator

## Overview
This package contains two Python scripts to generate daily 2×2 comparison maps showing wildfire impacts on air pollutants. Each script creates a PowerPoint presentation with 30 slides (one per day in June 2023).

---

## Scripts Included

### 1. **`create_daily_2x2_ppt_manual_scaling.py`**
- Uses **manually defined** color scale levels for each pollutant
- Best when you want **consistent color scales** across different days/runs
- Allows **precise control** over the color ranges
- Recommended for **publication-quality** figures

### 2. **`create_daily_2x2_ppt_auto_scaling.py`**
- **Automatically calculates** color scale levels from the data
- Best for **exploratory analysis** when you don't know the data range
- Uses smart rounding to create clean level values
- Saves time when analyzing new pollutants

---

## How to Use

### Step 1: Choose a Script
- For **consistent, publication-ready plots**: Use `create_daily_2x2_ppt_manual_scaling.py`
- For **quick exploration**: Use `create_daily_2x2_ppt_auto_scaling.py`

### Step 2: Edit Configuration (Top of Script)

Open the script in a text editor and modify these settings:

```python
# ============= USER CONFIGURATION =============
# Select pollutant
POLLUTANT = 'O3'  # Options: 'O3', 'PM25_TOT', 'CO', 'BENZENE', 'TOLUENE', 'PHENOL'

# Unit conversion
CONVERT_TO_UGM3 = False  # True: convert to μg/m³, False: keep native units

# Paths (update if needed)
BASE_DIR = r'D:/Raw_Data/CMAQ_Model/'
OUTPUT_DIR = r'C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\figures\daily_2x2_maps'

# Daily aggregation
DAILY_METHOD = 'mean'  # Options: 'mean' or 'max'
# ==============================================
```

### Step 3: Run the Script

**Option A - Command Line:**
```bash
cd C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis
python create_daily_2x2_ppt_manual_scaling.py
```

**Option B - Python Environment:**
```python
exec(open('create_daily_2x2_ppt_manual_scaling.py').read())
```

### Step 4: Find Your Output

The PowerPoint will be saved to:
```
C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\figures\daily_2x2_maps\
```

Filename format: `{POLLUTANT}_{UNITS}_daily_2x2_June2023.pptx`

Examples:
- `O3_ppb_daily_2x2_June2023.pptx`
- `O3_ugm3_daily_2x2_June2023.pptx`
- `PM25_ugm3_daily_2x2_June2023.pptx`

---

## Available Pollutants

| Pollutant Code | Display Name | Native Units | Molecular Weight |
|----------------|--------------|--------------|------------------|
| `O3`           | O₃           | ppb          | 48.00            |
| `PM25_TOT`     | PM₂.₅        | μg/m³        | N/A              |
| `CO`           | CO           | ppb          | 28.01            |
| `BENZENE`      | Benzene      | ppb          | 78.11            |
| `TOLUENE`      | Toluene      | ppb          | 92.14            |
| `PHENOL`       | Phenol       | ppb          | 94.11            |

---

## Understanding the Output

Each slide contains **4 panels**:

### Panel (a): Base Simulation
- CMAQ simulation **with fires**
- Shows total pollutant concentration
- Colormap: Viridis (perceptually uniform)

### Panel (b): No-Fire Scenario
- CMAQ simulation **without fires**
- Shows background concentration
- Colormap: Viridis (same scale as base)

### Panel (c): Fire Impact (Absolute)
- **Δ = Base − No Fire**
- Shows absolute change due to fires
- Units: Same as concentration (ppb or μg/m³)
- Colormap:
  - **RdBu_r** (Red-Blue) for O₃ (can be positive or negative)
  - **YlOrRd** (Yellow-Orange-Red) for PM₂.₅, CO, etc. (always positive)

### Panel (d): Fire Impact (Relative)
- **% Change = (Base − No Fire) / No Fire × 100**
- Shows percent change due to fires
- Units: Percent (%)
- Colormap: Same as Panel (c)

---

## Unit Conversion

### When `CONVERT_TO_UGM3 = False`:
- Gases (O₃, CO, Benzene, etc.): Displayed in **ppb**
- PM₂.₅: Displayed in **μg/m³** (already in μg/m³)

### When `CONVERT_TO_UGM3 = True`:
- **All pollutants** displayed in **μg/m³**
- Conversion formula for gases:
  ```
  μg/m³ = ppb × Air_Density × Molecular_Weight / 28.9628
  ```

---

## Customizing Color Scales (Manual Scaling Script Only)

To change color levels for a pollutant, edit the `POLLUTANT_CONFIG` dictionary:

```python
'O3': {
    'var_name': 'O3',
    'display_name': 'O₃',
    'native_units': 'ppb',
    'colormap_base': 'viridis',
    'colormap_delta': 'RdBu_r',
    'levels_base': [0, 10, 20, 30, 40, 50, 60, 70, 80],      # ← Edit these
    'levels_delta': [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20],   # ← Edit these
    'levels_percent': [-50, -20, -10, -5, 0, 5, 10, 20, 50], # ← Edit these
    'can_be_negative': True,
    'molecular_weight': 48.00,
},
```

### Tips:
- More levels = smoother gradients (but slower rendering)
- Fewer levels = clearer patterns (faster rendering)
- For delta plots with `can_be_negative = True`, use symmetric ranges (e.g., -10 to +10)

---

## Auto-Scaling Options (Auto Scaling Script Only)

```python
# Auto-scaling parameters
N_LEVELS = 9  # Number of contour levels (3-15 recommended)

USE_PERCENTILES = True  # True: use 2nd-98th percentile (robust to outliers)
                        # False: use min-max (sensitive to outliers)
```

### When to Use Percentiles:
- ✅ Data has extreme outliers
- ✅ You want smooth, consistent scales
- ✅ Publication-quality figures

### When to Use Min-Max:
- ✅ You want to see ALL data values
- ✅ No significant outliers in your data
- ✅ Exploratory analysis

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pptx'"
**Solution:** Install python-pptx:
```bash
pip install python-pptx
```

### Error: "FileNotFoundError: [file path]"
**Solution:** Check that `BASE_DIR` points to your CMAQ data folder:
```python
BASE_DIR = r'D:/Raw_Data/CMAQ_Model/'  # Update this path
```

### Error: "Unknown pollutant: XYZ"
**Solution:** Make sure `POLLUTANT` is one of the supported options. Check spelling and case:
```python
POLLUTANT = 'O3'  # Not 'o3' or 'ozone'
```

### Error: "ValueError: cannot reshape array"
**Solution:** Your data might not have 720 timesteps (30 days × 24 hours). Check:
```python
print(baseconc['O3'].shape)  # Should be (720, ROW, COL)
```

### Output folder not created
**Solution:** The script creates the folder automatically. Check permissions:
```python
OUTPUT_DIR = r'C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\figures\daily_2x2_maps'
```

---

## Example Workflow

### Scenario 1: Quick O₃ Analysis (ppb)
```python
POLLUTANT = 'O3'
CONVERT_TO_UGM3 = False
# Run auto_scaling script
```
**Output:** `O3_ppb_daily_2x2_June2023_auto.pptx`

---

### Scenario 2: O₃ Analysis (μg/m³)
```python
POLLUTANT = 'O3'
CONVERT_TO_UGM3 = True
# Run auto_scaling script
```
**Output:** `O3_ugm3_daily_2x2_June2023_auto.pptx`

---

### Scenario 3: PM₂.₅ Analysis for Publication
```python
POLLUTANT = 'PM25_TOT'
CONVERT_TO_UGM3 = False  # Already in μg/m³
# Run manual_scaling script
# Edit levels if needed for better visualization
```
**Output:** `PM25_ugm3_daily_2x2_June2023.pptx`

---

### Scenario 4: Compare All Pollutants
Run the script multiple times with different pollutants:
```python
# Run 1: O3
POLLUTANT = 'O3'
python create_daily_2x2_ppt_auto_scaling.py

# Run 2: PM2.5
POLLUTANT = 'PM25_TOT'
python create_daily_2x2_ppt_auto_scaling.py

# Run 3: CO
POLLUTANT = 'CO'
python create_daily_2x2_ppt_auto_scaling.py
```

All outputs saved to the same folder for easy comparison!

---

## Adding New Pollutants

To add a new pollutant (e.g., `NO2`), edit the `POLLUTANT_CONFIG` dictionary:

```python
'NO2': {
    'var_name': 'NO2',           # Variable name in NetCDF file
    'display_name': 'NO₂',       # Display name (with subscripts)
    'native_units': 'ppb',       # Native units
    'colormap_base': 'viridis',  # Colormap for base/nofire plots
    'colormap_delta': 'RdBu_r',  # Colormap for delta plots
    'can_be_negative': True,     # Can delta be negative?
    'molecular_weight': 46.01,   # Molecular weight (g/mol)

    # Only for manual scaling script:
    'levels_base': [0, 5, 10, 15, 20, 30, 40, 50, 60],
    'levels_delta': [-10, -5, -2, 0, 2, 5, 10, 20, 50],
    'levels_percent': [-50, -20, -10, 0, 10, 20, 50, 100, 200],
},
```

Then run:
```python
POLLUTANT = 'NO2'
```

---

## Script Performance

### Typical Runtime:
- **~5-10 minutes** for 30 daily plots (depending on your computer)
- Progress is printed to console

### Disk Space:
- Each PPT file: **~5-15 MB** (depends on map complexity)
- Temporary PNG files are automatically deleted

---

## Tips for Best Results

1. **Test with auto-scaling first** to understand your data range
2. **Then use manual-scaling** for final publication figures
3. **Keep color scales consistent** across related analyses
4. **Document your settings** (save a copy of the script for each analysis)
5. **Check the first few slides** before letting the full script run

---

## Questions?

If you encounter issues, check:
1. ✅ All required Python packages installed (`pyrsig`, `pycno`, `python-pptx`, etc.)
2. ✅ Data paths are correct (`BASE_DIR`)
3. ✅ Output directory has write permissions
4. ✅ CMAQ data files exist and are readable
5. ✅ Pollutant name matches exactly (case-sensitive)

---

**Happy analyzing! 🔥🗺️📊**
