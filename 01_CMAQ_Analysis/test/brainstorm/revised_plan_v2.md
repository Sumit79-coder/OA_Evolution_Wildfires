# CMAQ Wildfire Analysis - REVISED Architecture Plan v2

**Date:** 2025-11-18
**Revision:** Based on review of `CMAQ_Fire_PM25_O3_Analysis.ipynb`
**Status:** Updated after understanding actual workflow

---

## 🔍 Key Findings from Test Notebook Review

### Critical Corrections to Initial Plan:

1. **❌ WRONG: Daily NetCDF files**
   **✅ CORRECT: Monthly NetCDF files with TSTEP dimension**
   - Files contain entire month (June 2023): `COMBINE_ACONC_*_202306.nc`
   - TSTEP dimension has hourly/daily time steps
   - Current workflow: `.mean(dim='TSTEP')` for monthly average
   - **Goal: Extract individual days from TSTEP dimension**

2. **❌ WRONG: 2x2 plot layout**
   **✅ CORRECT: 1x4 horizontal panel layout**
   ```
   [Base] [No Fire] [Delta Absolute] [Delta %]
   ```

3. **❌ WRONG: PowerPoint as only output**
   **✅ CORRECT: Multiple output formats**
   - PNG files with timestamp: `YYYYMMDD_pollutant_maps.png`
   - Interactive HTML (Plotly): `YYYYMMDD_interactive_smoke_evolution.html`
   - CSV exports: smoke-masked data, binned data

4. **✅ CONFIRMED: Pollutant-specific handling**
   - PM2.5: Sequential colormap (YlOrRd), always positive delta
   - O3: Diverging colormap (RdBu_r), can be negative (NO titration!)
   - Zero line needed for O3 plots

5. **✅ NEW: Additional analysis outputs**
   - Smoke-masked scatter data (benzene > 10 ppt threshold)
   - Quantile-binned median trends
   - City location extractions
   - Mass ratio calculations (PM2.5/CO, O3/CO in g/g)

---

## 📊 Actual Data Structure

### Input Files (Monthly NetCDF):
```
D:\Raw_Data\CMAQ_Model\netcdffiles\
├── COMBINE_ACONC_cmaq6acracmm3_base_2023_12US4_202306.nc
│   └── Variables: PM25_TOT, O3, CO, BENZENE, TOLUENE, AIR_DENS
│   └── Dimensions: TSTEP (30 days × 24 hours), LAY, ROW, COL
└── COMBINE_ACONC_cmaq6acracmm3_nofire_2023_12US4_202306.nc
    └── Same structure
```

### What User Actually Wants:
1. **Analysis Script**: For each day in the month, extract daily data and compute fire impacts
2. **Plotting Script**: For each day, create 1x4 panel plots for selected pollutants
3. **Flexible pollutant selection**: Easy to switch between PM2.5, O3, CO, etc.
4. **Multiple outputs**: PNG files, interactive HTML, CSV data

---

## 🎯 Revised Solution Architecture

### Core Workflow:

```
Monthly NetCDF files (TSTEP dimension)
          ↓
[1. Extract Daily Data]
  - Select specific TSTEP index for each day
  - Compute: base, nofire, delta_abs, delta_percent
  - Calculate derived quantities (photochemical age, ratios)
          ↓
[2. Save Processed Data]
  - NetCDF: Daily gridded data (preserves spatial structure)
  - CSV: Smoke-masked point data, city extractions
          ↓
[3. Generate Plots]
  - 1×4 panel maps (base, nofire, delta, delta%)
  - Optional: Scatter plots (pollutant vs age)
  - Optional: Interactive Plotly HTML
          ↓
[4. Organize Outputs]
  - PNG files: figures/daily_plots/YYYY-MM-DD/
  - HTML: figures/interactive/
  - CSV: processed_data/timeseries/
  - Optional: Compile to PowerPoint
```

---

## 📁 Simplified Directory Structure

```
01_CMAQ_Analysis/
├── config/
│   ├── paths.yaml              # File paths
│   ├── pollutants.yaml         # Pollutant definitions (MW, colormaps, levels)
│   └── analysis_config.yaml   # Age calculation, smoke threshold
│
├── scripts/
│   ├── 01_extract_daily_data.py        # Extract days from monthly NetCDF
│   ├── 02_generate_daily_plots.py      # Create 1×4 panel plots
│   ├── 03_create_interactive_plots.py  # Plotly HTML plots
│   ├── 04_compile_to_ppt.py           # Optional: Assemble PowerPoint
│   │
│   └── utils/
│       ├── data_loader.py          # Load monthly NetCDF, select TSTEP
│       ├── calculator.py           # Fire impact calculations
│       ├── exporter.py             # Save NetCDF/CSV
│       ├── plotter_1x4.py          # 1×4 panel generator
│       ├── map_utils.py            # Map styling (pycno boundaries)
│       └── config_loader.py        # YAML config parser
│
├── processed_data/
│   ├── daily_gridded/          # NetCDF files per day/pollutant
│   │   ├── 2023-06-01_PM25.nc
│   │   ├── 2023-06-01_O3.nc
│   │   └── ...
│   ├── timeseries/             # CSV extractions
│   │   ├── smoke_masked_data.csv
│   │   ├── binned_data.csv
│   │   └── city_timeseries.csv
│   └── metadata/
│       └── processing_log.json
│
├── figures/
│   ├── daily_plots/
│   │   ├── 2023-06-01/
│   │   │   ├── PM25_1x4.png
│   │   │   ├── O3_1x4.png
│   │   │   └── CO_1x4.png
│   │   └── ...
│   ├── interactive/
│   │   └── 2023-06-01_smoke_evolution.html
│   └── presentations/
│       └── June2023_summary.pptx
│
├── notebooks/              # Keep existing exploratory notebooks
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration Files (YAML)

### `config/pollutants.yaml`

```yaml
pollutants:
  PM25:
    netcdf_var: 'PM25_TOT'
    display_name: 'PM$_{2.5}$'
    units: 'μg/m³'
    molecular_weight: null  # Already in mass units

    colormaps:
      base: 'viridis'
      nofire: 'viridis'
      delta: 'YlOrRd'        # Sequential (positive only)
      delta_percent: 'YlOrRd'

    plot_levels:
      base: [0, 10, 20, 30, 40, 50, 60, 70, 80]
      nofire: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      delta: [-0.5, 0, 0.5, 1, 2, 3, 4, 5, 10, 20]
      delta_percent: [-1, 0, 1, 2, 3, 4, 5, 10, 50, 100]

  O3:
    netcdf_var: 'O3'
    display_name: 'O$_3$'
    units: 'ppb'
    molecular_weight: 48.00

    colormaps:
      base: 'viridis'
      nofire: 'viridis'
      delta: 'RdBu_r'          # DIVERGING (can be negative!)
      delta_percent: 'RdBu_r'

    plot_levels:
      base: [0, 10, 20, 30, 40, 50, 60, 70, 80]
      nofire: [0, 10, 20, 30, 40, 50, 60, 70, 80]
      delta: [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
      delta_percent: [-50, -20, -10, -5, 0, 5, 10, 20, 50, 100]

    # Special handling for O3
    show_zero_line: true  # Add red dashed line at y=0

  CO:
    netcdf_var: 'CO'
    display_name: 'CO'
    units: 'ppb'
    molecular_weight: 28.01

    colormaps:
      base: 'viridis'
      nofire: 'viridis'
      delta: 'Purples'
      delta_percent: 'Purples'

    plot_levels:
      base: [0, 100, 200, 300, 400, 500, 600, 700, 800]
      nofire: [0, 50, 100, 150, 200, 250, 300, 350]
      delta: [0, 50, 100, 150, 200, 250, 300, 350]
      delta_percent: [-1, 0, 1, 2, 5, 10, 20, 50, 100]

  BENZENE:
    netcdf_var: 'BENZENE'
    display_name: 'Benzene'
    units: 'ppb'
    molecular_weight: 78.11

    colormaps:
      delta: 'Purples'

    plot_levels:
      delta: [0.01, 0.1, 0.5, 1, 2, 3, 4, 5]

  TOLUENE:
    netcdf_var: 'TOLUENE'
    display_name: 'Toluene'
    units: 'ppb'
    molecular_weight: 92.14

    colormaps:
      delta: 'Purples'

    plot_levels:
      delta: [0.01, 0.1, 0.5, 1, 2, 3, 4, 5]
```

### `config/analysis_config.yaml`

```yaml
input:
  # Monthly NetCDF files (contain TSTEP dimension)
  month_to_process: '202306'  # YYYYMM format

  # Time dimension handling
  tstep_frequency: 'hourly'  # or 'daily' depending on file
  extract_daily_mean: true   # Average all hours in a day

photochemical_age:
  benzene_toluene_ratio_initial: 2.27
  k_benzene: 1.2196e-12  # cm³/molec/s
  k_toluene: 5.9337e-12
  oh_concentration: 1.0e6  # molec/cm³
  smoke_threshold_benzene: 0.010  # ppb (10 ppt)

unit_conversions:
  air_density_var: 'AIR_DENS'  # kg/m³
  air_mw: 28.9628  # g/mol

smoke_masking:
  enable: true
  benzene_threshold: 0.010  # ppb

binning:
  enable: true
  quantiles: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  bin_variable: 'log_age'  # Bin on log(age) for even distribution

output:
  save_netcdf: true      # Daily gridded data
  save_csv: true         # Smoke-masked and binned data
  save_city_data: true   # Extract city locations
```

### `config/paths.yaml`

```yaml
raw_data:
  base_dir: 'D:/Raw_Data/CMAQ_Model'

  netcdf:
    base_simulation: 'netcdffiles/COMBINE_ACONC_cmaq6acracmm3_base_2023_12US4_202306.nc'
    nofire_simulation: 'netcdffiles/COMBINE_ACONC_cmaq6acracmm3_nofire_2023_12US4_202306.nc'

  meteorology:
    metcro2d: 'MOD3DATA_MET/METCRO2D.12US4.35L.230701'
    gridcro2d: 'MOD3DATA_MET/GRIDCRO2D.12US4.35L.230701'

  cities: 'cities.txt'

output:
  base_dir: 'C:/Users/smtku/OA_Evolution_Wildfires/01_CMAQ_Analysis'
  processed_data: 'processed_data'
  figures: 'figures'
  logs: 'logs'

plot_config:
  dpi: 300
  format: 'png'

  # 1×4 panel layout (horizontal)
  figure_size: [14, 1.8]  # inches (width, height)
  panel_spacing: 0.3

  # Panel titles
  panel_titles:
    base: 'June {pollutant}'
    nofire: 'No fire (Max: {max:.0f}, Min: {min:.0f})'
    delta_abs: '$\\Delta$C$_i$ (Max: {max:.1f}, Min: {min:.1e})'
    delta_percent: '%$\\Delta$C$_i$ (Max: {max:.1f}, Min: {min:.1e})'

  # Map styling
  hide_axes: true
  hide_ticks: true
  boundary_linewidth: 0.2
```

---

## 📊 Core Scripts

### `scripts/01_extract_daily_data.py`

```python
"""
Extract daily fire impact data from monthly CMAQ NetCDF files.

IMPORTANT: Input files contain TSTEP dimension (30 days × 24 hours).
This script extracts individual days and computes fire impacts.

Usage:
    # Process single day
    python 01_extract_daily_data.py --date 2023-06-07 --pollutants PM25 O3

    # Process entire month
    python 01_extract_daily_data.py --month 202306 --pollutants all

    # Specific date range
    python 01_extract_daily_data.py --date-range 2023-06-01 2023-06-30 --pollutants PM25 O3 CO
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd

from utils.config_loader import load_config
from utils.data_loader import CMAQMonthlyLoader
from utils.calculator import FireImpactCalculator
from utils.exporter import DataExporter

def main():
    parser = argparse.ArgumentParser(description='Extract daily CMAQ fire impact data')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--date-range', nargs=2, help='Start and end dates')
    parser.add_argument('--month', type=str, help='Month to process (YYYYMM)')
    parser.add_argument('--pollutants', nargs='+', default=['PM25'],
                       help='Pollutants: PM25 O3 CO BENZENE TOLUENE (or "all")')
    parser.add_argument('--output-dir', type=str, default='processed_data/daily_gridded')
    parser.add_argument('--export-csv', action='store_true', help='Export CSV timeseries')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Load configs
    config = load_config('config/analysis_config.yaml')
    pollutant_defs = load_config('config/pollutants.yaml')
    paths = load_config('config/paths.yaml')

    # Initialize loader (loads monthly NetCDF files)
    loader = CMAQMonthlyLoader(paths)
    calc = FireImpactCalculator(config, pollutant_defs)
    exporter = DataExporter(args.output_dir, config)

    # Determine dates to process
    if args.date:
        dates = [datetime.strptime(args.date, '%Y-%m-%d')]
    elif args.date_range:
        start = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        dates = pd.date_range(start, end, freq='D')
    elif args.month:
        # Process all days in month
        year = int(args.month[:4])
        month = int(args.month[4:6])
        dates = pd.date_range(f'{year}-{month:02d}-01', periods=30, freq='D')
    else:
        print("Error: Specify --date, --date-range, or --month")
        return

    # Determine pollutants
    if 'all' in args.pollutants:
        pollutants = list(pollutant_defs['pollutants'].keys())
    else:
        pollutants = args.pollutants

    print(f"📊 Processing {len(dates)} days, {len(pollutants)} pollutants")
    print(f"   Input:  {paths['raw_data']['base_dir']}")
    print(f"   Output: {args.output_dir}")

    # Process each day
    for date in dates:
        print(f"\n📅 {date.strftime('%Y-%m-%d')}")

        # Load data for this specific day from monthly file
        # (extracts TSTEP corresponding to this date)
        base_day = loader.load_day(date, scenario='base')
        nofire_day = loader.load_day(date, scenario='nofire')

        for pollutant in pollutants:
            print(f"   - {pollutant}")

            # Calculate fire impacts for this day
            results = calc.calculate_fire_impact(
                base_day, nofire_day, pollutant, date
            )

            # Results dict contains:
            # - base: Daily concentration (base scenario)
            # - nofire: Daily concentration (no-fire scenario)
            # - delta_abs: Absolute difference (base - nofire)
            # - delta_percent: Percent difference
            # - metadata: Processing info

            # Export to NetCDF (gridded)
            exporter.save_netcdf(results, pollutant, date)

            # Optional: Export CSV (smoke-masked points)
            if args.export_csv:
                exporter.save_csv(results, pollutant, date)

    print("\n✅ Extraction complete!")

if __name__ == '__main__':
    main()
```

### `scripts/02_generate_daily_plots.py`

```python
"""
Generate 1×4 panel plots from processed daily data.

Panel layout (horizontal):
┌────────┬──────────┬────────────┬──────────────┐
│  Base  │ No Fire  │ Delta (abs)│  Delta (%)   │
└────────┴──────────┴────────────┴──────────────┘

Usage:
    # Single day
    python 02_generate_daily_plots.py --date 2023-06-07 --pollutants PM25 O3

    # Date range
    python 02_generate_daily_plots.py --date-range 2023-06-01 2023-06-30 --pollutants all
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

from utils.config_loader import load_config
from utils.plotter_1x4 import OneFourPlotGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate daily 1×4 panel plots')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--date-range', nargs=2, help='Start and end dates')
    parser.add_argument('--pollutants', nargs='+', default=['PM25'])
    parser.add_argument('--input-dir', type=str, default='processed_data/daily_gridded')
    parser.add_argument('--output-dir', type=str, default='figures/daily_plots')
    parser.add_argument('--dpi', type=int, default=300)

    args = parser.parse_args()

    # Load configs
    pollutant_defs = load_config('config/pollutants.yaml')
    paths = load_config('config/paths.yaml')

    # Determine dates
    if args.date:
        dates = [datetime.strptime(args.date, '%Y-%m-%d')]
    elif args.date_range:
        start = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        dates = pd.date_range(start, end, freq='D')

    # Determine pollutants
    if 'all' in args.pollutants:
        pollutants = list(pollutant_defs['pollutants'].keys())
    else:
        pollutants = args.pollutants

    # Initialize plot generator
    plotter = OneFourPlotGenerator(paths['plot_config'], pollutant_defs)

    print(f"🎨 Generating plots for {len(dates)} days, {len(pollutants)} pollutants")

    # Generate plots
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        timestamp = date.strftime('%Y%m%d')
        print(f"\n📅 {date_str}")

        for pollutant in pollutants:
            # Load processed NetCDF
            data_file = Path(args.input_dir) / f"{date_str}_{pollutant}.nc"

            if not data_file.exists():
                print(f"   ⚠️  {pollutant}: Data file not found, skipping")
                continue

            ds = xr.open_dataset(data_file)

            # Generate 1×4 plot
            fig = plotter.create_1x4_panel(ds, pollutant, date)

            # Save figure
            output_dir = Path(args.output_dir) / date_str
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{pollutant}_1x4.png"

            fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
            print(f"   ✅ {pollutant}: {output_file}")

            # Also save with timestamp prefix (like in notebook)
            timestamp_file = output_dir / f"{timestamp}_{pollutant}_maps.png"
            fig.savefig(timestamp_file, dpi=args.dpi, bbox_inches='tight')

    print("\n✅ Plot generation complete!")

if __name__ == '__main__':
    main()
```

---

## 🔑 Key Utility Modules

### `utils/data_loader.py` - REVISED

```python
"""
Load monthly CMAQ NetCDF files and extract daily slices.

Key difference from original plan:
- Input files are MONTHLY (not daily)
- TSTEP dimension contains 30 days × 24 hours (or daily averages)
- This module extracts specific days from the TSTEP dimension
"""

import pyrsig
import xarray as xr
from pathlib import Path
from datetime import datetime

class CMAQMonthlyLoader:
    """Load monthly CMAQ files and extract daily data."""

    def __init__(self, paths_config):
        self.config = paths_config
        self.base_dir = Path(self.config['raw_data']['base_dir'])

        # Load monthly NetCDF files (contains entire month)
        base_file = self.base_dir / self.config['raw_data']['netcdf']['base_simulation']
        nofire_file = self.base_dir / self.config['raw_data']['netcdf']['nofire_simulation']

        print(f"📂 Loading monthly NetCDF files...")
        print(f"   Base:   {base_file}")
        print(f"   NoFire: {nofire_file}")

        self.ds_base = pyrsig.open_ioapi(str(base_file))
        self.ds_nofire = pyrsig.open_ioapi(str(nofire_file))

        print(f"   ✅ Loaded. TSTEP dimension: {len(self.ds_base.TSTEP)} time steps")

    def load_day(self, date, scenario='base'):
        """
        Extract data for a specific day from monthly file.

        Parameters:
        -----------
        date : datetime
            Date to extract
        scenario : str
            'base' or 'nofire'

        Returns:
        --------
        xarray.Dataset
            Data for the specified day
        """
        ds = self.ds_base if scenario == 'base' else self.ds_nofire

        # Find TSTEP index for this date
        # (Implementation depends on how TSTEP is encoded in the file)
        # Option 1: If TSTEP has datetime coordinates
        try:
            ds_day = ds.sel(TSTEP=date, method='nearest')
        except:
            # Option 2: If TSTEP is indexed by day number
            day_of_month = date.day - 1  # 0-indexed
            ds_day = ds.isel(TSTEP=day_of_month)

        # If hourly data, average over hours
        if 'TSTEP' in ds_day.dims:
            ds_day = ds_day.mean(dim='TSTEP')

        return ds_day

    def load_met(self):
        """Load meteorology file for unit conversions."""
        met_file = self.base_dir / self.config['raw_data']['meteorology']['metcro2d']
        return pyrsig.open_ioapi(str(met_file))

    def load_grid(self):
        """Load grid definition for map projections."""
        grid_file = self.base_dir / self.config['raw_data']['meteorology']['gridcro2d']
        return pyrsig.open_ioapi(str(grid_file))
```

### `utils/plotter_1x4.py` - NEW

```python
"""
Generate 1×4 horizontal panel plots.

Layout:
┌────────┬──────────┬────────────┬──────────────┐
│  Base  │ No Fire  │ Delta (abs)│  Delta (%)   │
└────────┴──────────┴────────────┴──────────────┘
"""

import matplotlib.pyplot as plt
import numpy as np

class OneFourPlotGenerator:
    """Generate 1×4 panel maps."""

    def __init__(self, plot_config, pollutant_defs):
        self.config = plot_config
        self.pollutant_defs = pollutant_defs

    def create_1x4_panel(self, dataset, pollutant, date):
        """
        Create 1×4 panel plot.

        Parameters:
        -----------
        dataset : xarray.Dataset
            Must contain: base, nofire, delta_abs, delta_percent
        pollutant : str
            Pollutant name (PM25, O3, CO, etc.)
        date : datetime
            Date for plot title

        Returns:
        --------
        matplotlib.figure.Figure
        """
        p_config = self.pollutant_defs['pollutants'][pollutant]

        # Create figure with 1 row, 4 columns
        fig, axes = plt.subplots(
            1, 4,
            figsize=self.config['figure_size'],
            dpi=self.config['dpi']
        )
        plt.subplots_adjust(wspace=self.config['panel_spacing'])

        # Panel 0: Base simulation
        ax = axes[0]
        conc = dataset['base']
        levels = p_config['plot_levels']['base']
        cmap = p_config['colormaps']['base']

        pv = conc.plot(ax=ax, cmap=cmap, levels=levels, add_colorbar=True)
        title = self.config['panel_titles']['base'].format(
            pollutant=p_config['display_name']
        )
        self._style_panel(ax, pv, title, p_config['units'])

        # Panel 1: No-fire simulation
        ax = axes[1]
        conc = dataset['nofire']
        levels = p_config['plot_levels']['nofire']
        cmap = p_config['colormaps']['nofire']

        pv = conc.plot(ax=ax, cmap=cmap, levels=levels, add_colorbar=True)
        title = self.config['panel_titles']['nofire'].format(
            max=conc.max().values,
            min=conc.min().values
        )
        self._style_panel(ax, pv, title, p_config['units'])

        # Panel 2: Delta (absolute)
        ax = axes[2]
        conc = dataset['delta_abs']
        levels = p_config['plot_levels']['delta']
        cmap = p_config['colormaps']['delta']

        pv = conc.plot(ax=ax, cmap=cmap, levels=levels, add_colorbar=True)
        title = self.config['panel_titles']['delta_abs'].format(
            max=conc.max().values,
            min=conc.min().values
        )
        self._style_panel(ax, pv, title, p_config['units'])

        # Panel 3: Delta (percent)
        ax = axes[3]
        conc = dataset['delta_percent']
        levels = p_config['plot_levels']['delta_percent']
        cmap = p_config['colormaps']['delta_percent']

        pv = conc.plot(ax=ax, cmap=cmap, levels=levels, add_colorbar=True)
        title = self.config['panel_titles']['delta_percent'].format(
            max=conc.max().values,
            min=conc.min().values
        )
        self._style_panel(ax, pv, title, '%')

        return fig

    def _style_panel(self, ax, plot_obj, title, colorlabel):
        """Apply consistent styling to panel."""
        plot_obj.colorbar.set_label(colorlabel)
        ax.set_title(title, fontsize=10)

        if self.config['hide_axes']:
            ax.set_xlabel('')
            ax.set_ylabel('')

        if self.config['hide_ticks']:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add state boundaries (using pycno)
        # TODO: Implement boundary drawing
```

---

## 📈 Additional Scripts (Optional)

### `scripts/03_create_interactive_plots.py`

Generate Plotly interactive HTML plots (like in the test notebook):
- PM2.5 vs age
- O3 vs age
- PM2.5/CO vs age
- O3/CO vs age
- City markers with hover tooltips

### `scripts/04_compile_to_ppt.py`

Assemble daily PNG plots into PowerPoint presentation.

---

## 🎨 Plot Examples

### 1×4 Panel for PM2.5:
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ June PM2.5      │ No Fire         │ ΔC_i            │ %ΔC_i           │
│ (Base)          │ (Max: 10)       │ (Max: 20.5)     │ (Max: 150%)     │
│                 │                 │                 │                 │
│ [viridis map]   │ [viridis map]   │ [YlOrRd map]    │ [YlOrRd map]    │
│ 0-80 μg/m³      │ 0-10 μg/m³      │ 0-20 μg/m³      │ 0-100%          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### 1×4 Panel for O3 (note diverging colormap):
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ June O3         │ No Fire         │ ΔO3             │ %ΔO3            │
│ (Base)          │ (Max: 75)       │ (Range: -5 to 5)│ (Range: -20 to  │
│                 │                 │                 │  +50%)          │
│ [viridis map]   │ [viridis map]   │ [RdBu_r map]    │ [RdBu_r map]    │
│ 0-80 ppb        │ 0-80 ppb        │ -10 to +20 ppb  │ -50 to +100%    │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

---

## ⚙️ Usage Examples

### Scenario 1: Extract and plot single day
```bash
# Step 1: Extract daily data
python scripts/01_extract_daily_data.py \
    --date 2023-06-07 \
    --pollutants PM25 O3 CO

# Step 2: Generate plots
python scripts/02_generate_daily_plots.py \
    --date 2023-06-07 \
    --pollutants PM25 O3 CO
```

### Scenario 2: Process entire month
```bash
# Extract all days in June 2023
python scripts/01_extract_daily_data.py \
    --month 202306 \
    --pollutants all \
    --export-csv

# Generate all plots
python scripts/02_generate_daily_plots.py \
    --date-range 2023-06-01 2023-06-30 \
    --pollutants all
```

### Scenario 3: Create interactive plots
```bash
# Generate Plotly HTML
python scripts/03_create_interactive_plots.py \
    --date 2023-06-07 \
    --pollutants PM25 O3
```

---

## 📊 Output File Naming

Following the notebook's convention:

### PNG Plots:
```
figures/daily_plots/2023-06-07/
├── 20230607_PM25_maps.png          # Timestamp prefix
├── 20230607_O3_maps.png
├── PM25_1x4.png                    # Descriptive name
└── O3_1x4.png
```

### Interactive HTML:
```
figures/interactive/
└── 20230607_interactive_smoke_evolution.html
```

### CSV Exports:
```
processed_data/timeseries/
├── 20230607_smoke_masked_data.csv
└── 20230607_binned_data.csv
```

---

## 🔍 Key Differences from Original Plan

| Aspect | Original Plan | REVISED Plan |
|--------|--------------|--------------|
| **Input data** | Daily NetCDF files | Monthly NetCDF with TSTEP dimension |
| **Data extraction** | Direct file load | Select from TSTEP dimension |
| **Plot layout** | 2×2 grid | 1×4 horizontal panels |
| **File naming** | Date prefix optional | Timestamp prefix standard |
| **CSV exports** | Optional | Standard output |
| **Interactive plots** | Not mentioned | Plotly HTML files |
| **O3 handling** | Same as PM2.5 | Diverging colormap, zero line |

---

## ✅ Validation Checklist

Before finalizing implementation:

- [ ] Confirm TSTEP dimension structure in NetCDF files
- [ ] Test TSTEP indexing (datetime vs integer)
- [ ] Verify hourly vs daily data in TSTEP
- [ ] Confirm colormap choices with user
- [ ] Test pycno boundary overlays
- [ ] Validate CSV export format preferences
- [ ] Check PowerPoint requirement (still needed?)

---

## 🎯 Discussion Points

### Questions for User:

1. **TSTEP dimension details:**
   - How many time steps in the monthly file? (720 hours or 30 days?)
   - Is TSTEP indexed by datetime or integers?
   - Should we average hours within each day?

2. **Plot preferences:**
   - Confirm 1×4 horizontal layout (not 2×2)?
   - Keep timestamp prefix in filenames?
   - What title format for each panel?

3. **Output priorities:**
   - NetCDF for processed data: Yes/No?
   - CSV exports: Which formats needed?
   - Interactive HTML: For all days or just summaries?
   - PowerPoint: Still required, or just PNG files?

4. **Pollutants to prioritize:**
   - Start with PM2.5, O3, CO?
   - Add Benzene, Toluene later?
   - Photochemical age maps?

5. **Processing scope:**
   - All 30 days of June?
   - Or specific days of interest?

6. **Additional analysis:**
   - Smoke-masked scatter plots (pollutant vs age)?
   - City time series?
   - Statistical summaries?

---

## 📚 Next Steps

1. **Review this revised plan** - Confirm understanding of workflow
2. **Clarify TSTEP dimension** - Check NetCDF file structure
3. **Prioritize features** - Essential vs nice-to-have
4. **Start implementation** - Build core modules first
5. **Test with one day** - Validate entire pipeline

---

**This revised plan accurately reflects the actual workflow shown in the test notebook!**
