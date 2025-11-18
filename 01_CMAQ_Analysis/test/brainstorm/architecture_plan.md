# CMAQ Wildfire Analysis - Software Architecture Plan

**Date:** 2025-11-18
**Author:** Senior Software Development Plan
**Project:** OA Evolution Wildfires - CMAQ Analysis Automation

---

## 🏗️ Executive Summary

This document outlines a comprehensive, enterprise-grade architecture for transforming Jupyter notebook-based CMAQ wildfire analysis into production-ready, modular Python scripts with automated data processing and visualization pipelines.

---

## 📋 Current State Analysis

### What We Found

**Strengths:**
- ✅ Well-documented Jupyter notebooks with CMAQ wildfire analysis
- ✅ Two simulation types: base (with fires) vs no-fire scenarios
- ✅ Focus pollutants: PM2.5, O3, CO, Benzene, Toluene
- ✅ Analysis includes: delta calculations, photochemical age, spatial maps, temporal evolution
- ✅ 2x2 panel plots: base, no-fire, absolute delta, relative delta (%)

**Current Limitations:**
- ❌ Analysis code embedded in notebooks (not reusable)
- ❌ Manual pollutant selection and processing
- ❌ No automated daily data extraction
- ❌ No batch plotting capabilities
- ❌ Hard-coded paths and parameters

### Data Sources

**Raw Data Location:** `D:\Raw_Data\CMAQ_Model\netcdffiles\`
- `COMBINE_ACONC_cmaq6acracmm3_base_2023_12US4_202306.nc`
- `COMBINE_ACONC_cmaq6acracmm3_nofire_2023_12US4_202306.nc`
- `COMBINE_ACONC_cmaq55cracmm2_base_2023_12US4_202306.nc`
- `COMBINE_ACONC_cmaq55cracmm2_nofire_2023_12US4_202306.nc`

**Meteorology Data:** `D:\Raw_Data\CMAQ_Model\MOD3DATA_MET\`
- `METCRO2D.12US4.35L.230701`
- `GRIDCRO2D.12US4.35L.230701`

**Analysis Workspace:** `C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\`

---

## 🎯 Proposed Solution: Enterprise-Grade Architecture

### Core Design Principles

1. **Separation of Concerns**: Data → Processing → Analysis → Visualization
2. **Configuration-Driven**: YAML configs for all parameters (no hard-coding)
3. **Modularity**: Reusable components with clear interfaces
4. **Scalability**: Handle multiple days, pollutants, domains in parallel
5. **Maintainability**: Clean code, comprehensive documentation
6. **Error Handling**: Robust validation and logging
7. **Performance**: Lazy loading, chunking for large datasets
8. **Extensibility**: Easy to add new pollutants, plot types, analysis methods
9. **Testability**: Unit tests for critical functions
10. **Reproducibility**: Version control, metadata tracking

---

## 📁 Proposed Directory Structure

```
OA_Evolution_Wildfires/
└── 01_CMAQ_Analysis/
    ├── config/                              # 🔧 Configuration files
    │   ├── paths.yaml                       # Data paths, output directories
    │   ├── pollutants.yaml                  # Pollutant definitions (MW, units, thresholds)
    │   ├── analysis_config.yaml             # Analysis parameters
    │   └── plot_config.yaml                 # Plot styling, colormaps, levels
    │
    ├── scripts/
    │   ├── 01_analysis/                     # 📊 Data extraction & processing
    │   │   ├── extract_daily_data.py        # CLI tool for daily data extraction
    │   │   ├── batch_process.py             # Batch processing for date ranges
    │   │   └── utils/
    │   │       ├── __init__.py
    │   │       ├── data_loader.py           # NetCDF loading with pyrsig
    │   │       ├── calculator.py            # Delta, age, ratio calculations
    │   │       ├── exporter.py              # Save to NetCDF/CSV/Parquet
    │   │       └── validators.py            # Input validation
    │   │
    │   ├── 02_plotting/                     # 📈 Visualization & export
    │   │   ├── generate_daily_plots.py      # CLI tool for 2x2 plots
    │   │   ├── batch_plot.py                # Batch plot generation
    │   │   ├── create_ppt.py                # Assemble PowerPoint deck
    │   │   └── utils/
    │   │       ├── __init__.py
    │   │       ├── plot_2x2.py              # 2x2 panel generator
    │   │       ├── map_utils.py             # Map projection, boundaries
    │   │       ├── colormap_registry.py     # Pollutant-specific colormaps
    │   │       └── ppt_builder.py           # python-pptx utilities
    │   │
    │   ├── utils/                           # 🛠️ Shared utilities
    │   │   ├── __init__.py
    │   │   ├── config_loader.py             # YAML config parser
    │   │   ├── logger.py                    # Centralized logging
    │   │   └── parallel.py                  # Multiprocessing helpers
    │   │
    │   └── run_pipeline.py                  # 🚀 Master orchestration script
    │
    ├── processed_data/                      # 💾 Processed outputs
    │   ├── daily/                           # Daily gridded data (NetCDF)
    │   │   ├── 2023-06-01_PM25.nc
    │   │   ├── 2023-06-01_O3.nc
    │   │   └── ...
    │   ├── timeseries/                      # Time series at city locations (CSV)
    │   │   └── cities_timeseries.csv
    │   └── metadata/                        # Processing logs & metadata
    │       └── processing_log.json
    │
    ├── figures/                             # 🖼️ Plots & presentations
    │   ├── daily_plots/                     # Individual day PNGs
    │   │   ├── 2023-06-01/
    │   │   │   ├── PM25_2x2.png
    │   │   │   ├── O3_2x2.png
    │   │   │   └── CO_2x2.png
    │   │   └── ...
    │   ├── presentations/                   # PowerPoint outputs
    │   │   └── June2023_WildfireSummary.pptx
    │   └── interactive/                     # Plotly HTML files
    │       └── PM25_interactive.html
    │
    ├── notebooks/                           # 📓 Exploratory analysis (keep existing)
    │   ├── Pye2025_FirePMevolution.ipynb
    │   ├── 01_data_exploration.ipynb
    │   └── 02_comprehensive_fire_analysis.ipynb
    │
    ├── tests/                               # ✅ Unit tests
    │   ├── test_calculator.py
    │   ├── test_data_loader.py
    │   └── test_plot_generator.py
    │
    ├── requirements.txt                     # Python dependencies
    ├── environment.yaml                     # Conda environment
    ├── README.md                            # Documentation
    └── .gitignore
```

---

## 🔧 Configuration System

### 1. Pollutant Definitions (`config/pollutants.yaml`)

```yaml
pollutants:
  PM25:
    netcdf_var: 'PM25_TOT'
    display_name: 'PM₂.₅'
    units: 'μg/m³'
    molecular_weight: null  # Already in mass units
    colormaps:
      base: 'viridis'
      delta: 'YlOrRd'
    plot_levels:
      base: [0, 10, 20, 30, 40, 50, 60, 70, 80]
      nofire: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      delta: [-0.5, 0, 0.5, 1, 2, 3, 4, 5, 10, 20]
      delta_percent: [-1, 0, 1, 2, 3, 4, 5, 10, 50, 100]

  O3:
    netcdf_var: 'O3'
    display_name: 'O₃'
    units: 'ppb'
    molecular_weight: 48.00
    colormaps:
      base: 'viridis'
      delta: 'RdBu_r'  # Diverging (can be negative!)
    plot_levels:
      base: [0, 10, 20, 30, 40, 50, 60, 70, 80]
      delta: [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
      delta_percent: [-50, -20, -10, -5, 0, 5, 10, 20, 50, 100]

  CO:
    netcdf_var: 'CO'
    display_name: 'CO'
    units: 'ppb'
    molecular_weight: 28.01
    colormaps:
      base: 'viridis'
      delta: 'Purples'
    plot_levels:
      base: [0, 100, 200, 300, 400, 500, 600, 700, 800]
      delta: [0, 50, 100, 150, 200, 250, 300, 350]

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

### 2. Analysis Configuration (`config/analysis_config.yaml`)

```yaml
analysis:
  date_range:
    start: '2023-06-01'
    end: '2023-06-30'

  temporal_aggregation: 'daily'  # daily, hourly, monthly

  photochemical_age:
    benzene_toluene_ratio_initial: 2.27
    k_benzene: 1.2196e-12  # cm³/molec/s
    k_toluene: 5.9337e-12
    oh_concentration: 1.0e6  # molec/cm³
    smoke_threshold_benzene: 0.010  # ppb

  unit_conversions:
    air_density_var: 'AIR_DENS'  # kg/m³
    air_mw: 28.9628  # g/mol

  output_formats:
    - 'netcdf'  # Gridded data (preserves spatial structure)
    - 'csv'     # City timeseries

  parallel:
    enable: true
    n_workers: 4
```

### 3. Path Configuration (`config/paths.yaml`)

```yaml
paths:
  raw_data:
    base_dir: 'D:\Raw_Data\CMAQ_Model'
    netcdf_dir: 'D:\Raw_Data\CMAQ_Model\netcdffiles'
    met_dir: 'D:\Raw_Data\CMAQ_Model\MOD3DATA_MET'

    # Simulation files
    base_simulation: 'COMBINE_ACONC_cmaq6acracmm3_base_2023_12US4_202306.nc'
    nofire_simulation: 'COMBINE_ACONC_cmaq6acracmm3_nofire_2023_12US4_202306.nc'

    # Meteorology files
    metcro2d: 'METCRO2D.12US4.35L.230701'
    gridcro2d: 'GRIDCRO2D.12US4.35L.230701'

    # Geographic overlays
    boundaries: 'overlays\MWDB_Coasts_NA_3.cnob'
    cities: 'cities.txt'

  output:
    base_dir: 'C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis'
    processed_data: 'processed_data'
    figures: 'figures'
    logs: 'logs'
```

### 4. Plot Configuration (`config/plot_config.yaml`)

```yaml
plotting:
  figure:
    dpi: 300
    format: 'png'
    single_panel_size: [3.3, 2.25]  # inches
    four_panel_size: [14, 1.8]

  layout:
    2x2:
      panels: ['base', 'nofire', 'delta_abs', 'delta_percent']
      titles:
        base: 'Base (with fires)'
        nofire: 'No Fire'
        delta_abs: 'ΔC (fire impact)'
        delta_percent: '%ΔC'
      wspace: 0.3  # Horizontal spacing
      hspace: 0.3  # Vertical spacing

  map:
    projection: 'lcc'  # Read from CMAQ files
    boundaries:
      type: 'cnob'  # pycno boundary file
      linewidth: 0.2
      color: 'black'
    hide_axes: true
    hide_ticks: true

  powerpoint:
    template: null  # Use default blank
    slide_layout: 'Title and Content'
    title_format: '{pollutant} - {date}'
    image_position:
      left: 0.5  # inches from left
      top: 1.5   # inches from top
      width: 9.0 # inches
      height: 6.0
```

---

## 📊 Analysis Scripts

### Main CLI Tool: `scripts/01_analysis/extract_daily_data.py`

```python
"""
Extract daily pollutant data from CMAQ simulations.

Usage:
    python extract_daily_data.py --date 2023-06-01 --pollutants PM25 O3 CO
    python extract_daily_data.py --date-range 2023-06-01 2023-06-30 --pollutants all
    python extract_daily_data.py --config custom_config.yaml
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

from utils.config_loader import load_config
from utils.data_loader import CMAQDataLoader
from utils.calculator import FireImpactCalculator
from utils.exporter import DataExporter
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Extract daily CMAQ data')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--date-range', nargs=2, help='Start and end dates')
    parser.add_argument('--pollutants', nargs='+', default=['PM25'],
                       help='Pollutants to process (or "all")')
    parser.add_argument('--config', type=str, default='config/analysis_config.yaml')
    parser.add_argument('--output-dir', type=str, default='processed_data/daily')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(level='DEBUG' if args.verbose else 'INFO')

    # Load configurations
    config = load_config(args.config)
    pollutant_defs = load_config('config/pollutants.yaml')
    paths = load_config('config/paths.yaml')

    # Determine dates to process
    if args.date:
        dates = [datetime.strptime(args.date, '%Y-%m-%d')]
    elif args.date_range:
        start = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        dates = [start + timedelta(days=x) for x in range((end-start).days + 1)]
    else:
        dates = get_dates_from_config(config)

    # Determine pollutants
    if 'all' in args.pollutants:
        pollutants = list(pollutant_defs['pollutants'].keys())
    else:
        pollutants = args.pollutants

    logger.info(f"Processing {len(dates)} days, {len(pollutants)} pollutants")

    # Initialize processors
    loader = CMAQDataLoader(paths)
    calculator = FireImpactCalculator(config, pollutant_defs)
    exporter = DataExporter(args.output_dir, config)

    # Process each date
    for date in dates:
        logger.info(f"Processing {date.strftime('%Y-%m-%d')}")

        # Load base and nofire data for this date
        base_data = loader.load_day(date, scenario='base')
        nofire_data = loader.load_day(date, scenario='nofire')

        for pollutant in pollutants:
            logger.info(f"  - {pollutant}")

            # Calculate fire impacts
            results = calculator.calculate_fire_impact(
                base_data, nofire_data, pollutant, date
            )

            # results contains: base, nofire, delta_abs, delta_percent, metadata

            # Export
            exporter.save(results, pollutant, date)

    logger.info("✅ Processing complete!")

if __name__ == '__main__':
    main()
```

### Key Utility Modules

#### `scripts/01_analysis/utils/data_loader.py`

```python
"""
CMAQ NetCDF data loading utilities.

Features:
- Load CMAQ concentration files with pyrsig
- Filter by date/time
- Handle multiple scenarios (base, nofire)
- Load meteorology and grid files
"""

import pyrsig
import xarray as xr
from pathlib import Path
from datetime import datetime

class CMAQDataLoader:
    def __init__(self, paths_config):
        self.config = paths_config
        self.base_file = Path(self.config['raw_data']['netcdf_dir']) / \
                         self.config['raw_data']['base_simulation']
        self.nofire_file = Path(self.config['raw_data']['netcdf_dir']) / \
                           self.config['raw_data']['nofire_simulation']

    def load_day(self, date, scenario='base'):
        """Load data for a single day."""
        file_path = self.base_file if scenario == 'base' else self.nofire_file

        # Load with pyrsig (handles IOAPI format)
        ds = pyrsig.open_ioapi(str(file_path))

        # Filter to specific date
        # (Implementation depends on how TSTEP is encoded)
        ds_day = ds.sel(TSTEP=date)  # Adjust based on actual time coordinate

        return ds_day

    def load_met(self):
        """Load meteorology data for unit conversions."""
        met_path = Path(self.config['raw_data']['met_dir']) / \
                   self.config['raw_data']['metcro2d']
        return pyrsig.open_ioapi(str(met_path))

    def load_grid(self):
        """Load grid definition for map projections."""
        grid_path = Path(self.config['raw_data']['met_dir']) / \
                    self.config['raw_data']['gridcro2d']
        return pyrsig.open_ioapi(str(grid_path))
```

#### `scripts/01_analysis/utils/calculator.py`

```python
"""
Fire impact calculations: deltas, unit conversions, photochemical age.
"""

import numpy as np
import xarray as xr

class FireImpactCalculator:
    def __init__(self, analysis_config, pollutant_defs):
        self.config = analysis_config
        self.pollutant_defs = pollutant_defs

    def calculate_fire_impact(self, base_data, nofire_data, pollutant, date):
        """
        Calculate fire impacts for a pollutant.

        Returns:
            dict with keys: base, nofire, delta_abs, delta_percent, metadata
        """
        var_name = self.pollutant_defs['pollutants'][pollutant]['netcdf_var']

        # Extract species
        base = base_data[var_name]
        nofire = nofire_data[var_name]

        # Calculate deltas
        delta_abs = base - nofire
        delta_percent = (delta_abs / base) * 100

        # Unit conversion if needed
        if self.pollutant_defs['pollutants'][pollutant]['molecular_weight']:
            # Convert ppb to μg/m³
            mw = self.pollutant_defs['pollutants'][pollutant]['molecular_weight']
            air_dens = base_data[self.config['analysis']['unit_conversions']['air_density_var']]
            air_mw = self.config['analysis']['unit_conversions']['air_mw']

            delta_abs_ugm3 = delta_abs * air_dens * mw / air_mw
        else:
            delta_abs_ugm3 = delta_abs  # Already in mass units

        return {
            'base': base,
            'nofire': nofire,
            'delta_abs': delta_abs,
            'delta_percent': delta_percent,
            'delta_abs_ugm3': delta_abs_ugm3,
            'metadata': {
                'date': date.strftime('%Y-%m-%d'),
                'pollutant': pollutant,
                'units': self.pollutant_defs['pollutants'][pollutant]['units']
            }
        }

    def calculate_photochemical_age(self, benzene_delta, toluene_delta):
        """Calculate smoke age from benzene/toluene ratio."""
        bz_tol_initial = self.config['analysis']['photochemical_age']['benzene_toluene_ratio_initial']
        k_bz = self.config['analysis']['photochemical_age']['k_benzene']
        k_tol = self.config['analysis']['photochemical_age']['k_toluene']
        oh_conc = self.config['analysis']['photochemical_age']['oh_concentration']

        ratio = benzene_delta / toluene_delta
        age_days = (np.log(ratio / bz_tol_initial) /
                   ((k_tol - k_bz) * oh_conc) /
                   3600 / 24)

        return age_days
```

#### `scripts/01_analysis/utils/exporter.py`

```python
"""
Export processed data to NetCDF, CSV, or Parquet.
"""

import xarray as xr
import pandas as pd
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir)
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results, pollutant, date):
        """Save processing results."""
        date_str = date.strftime('%Y-%m-%d')

        # Save as NetCDF (preserves spatial structure)
        if 'netcdf' in self.config['analysis']['output_formats']:
            output_file = self.output_dir / f"{date_str}_{pollutant}.nc"

            # Combine into dataset
            ds = xr.Dataset({
                'base': results['base'],
                'nofire': results['nofire'],
                'delta_abs': results['delta_abs'],
                'delta_percent': results['delta_percent']
            })

            # Add metadata
            ds.attrs['date'] = date_str
            ds.attrs['pollutant'] = pollutant
            ds.attrs['processing_date'] = pd.Timestamp.now().isoformat()

            ds.to_netcdf(output_file)

        # TODO: Add CSV export for city locations
```

---

## 📈 Plotting Scripts

### Main CLI Tool: `scripts/02_plotting/generate_daily_plots.py`

```python
"""
Generate 2x2 panel plots from processed data.

Usage:
    python generate_daily_plots.py --date 2023-06-01 --pollutants PM25 O3
    python generate_daily_plots.py --date-range 2023-06-01 2023-06-30 --pollutants all
    python generate_daily_plots.py --date 2023-06-01 --pollutants PM25 --to-ppt
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

from utils.config_loader import load_config
from utils.plot_2x2 import TwoByTwoPlotGenerator
from utils.ppt_builder import PowerPointBuilder
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Generate daily plots')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--date-range', nargs=2, help='Start and end dates')
    parser.add_argument('--pollutants', nargs='+', default=['PM25'])
    parser.add_argument('--input-dir', type=str, default='processed_data/daily')
    parser.add_argument('--output-dir', type=str, default='figures/daily_plots')
    parser.add_argument('--to-ppt', action='store_true', help='Add to PowerPoint')
    parser.add_argument('--ppt-output', type=str, default='figures/presentations/output.pptx')
    parser.add_argument('--config', type=str, default='config/plot_config.yaml')

    args = parser.parse_args()
    logger = setup_logger()

    # Load configs
    plot_config = load_config(args.config)
    pollutant_defs = load_config('config/pollutants.yaml')

    # Determine dates
    if args.date:
        dates = [datetime.strptime(args.date, '%Y-%m-%d')]
    elif args.date_range:
        start = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        dates = [start + timedelta(days=x) for x in range((end-start).days + 1)]

    # Initialize generators
    plot_gen = TwoByTwoPlotGenerator(plot_config, pollutant_defs)

    if args.to_ppt:
        ppt_builder = PowerPointBuilder(args.ppt_output, plot_config)

    # Generate plots
    for date in dates:
        logger.info(f"Plotting {date.strftime('%Y-%m-%d')}")

        for pollutant in args.pollutants:
            # Load processed data
            data_file = Path(args.input_dir) / f"{date.strftime('%Y-%m-%d')}_{pollutant}.nc"

            # Generate 2x2 plot
            fig, axes = plot_gen.create_2x2(data_file, pollutant, date)

            # Save PNG
            output_path = Path(args.output_dir) / date.strftime('%Y-%m-%d') / f"{pollutant}_2x2.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=plot_config['figure']['dpi'], bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")

            # Add to PowerPoint if requested
            if args.to_ppt:
                ppt_builder.add_slide(
                    title=f"{pollutant} - {date.strftime('%Y-%m-%d')}",
                    image_path=output_path
                )

    if args.to_ppt:
        ppt_builder.save()
        logger.info(f"✅ PowerPoint saved: {args.ppt_output}")

if __name__ == '__main__':
    main()
```

### Key Plotting Utilities

#### `scripts/02_plotting/utils/plot_2x2.py`

```python
"""
Generate 2x2 panel plots.

Layout:
┌─────────────┬─────────────┐
│ Base        │ No Fire     │
├─────────────┼─────────────┤
│ Delta (abs) │ Delta (%)   │
└─────────────┴─────────────┘
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path

from .map_utils import setup_map_projection, add_boundaries
from .colormap_registry import get_colormap, get_levels

class TwoByTwoPlotGenerator:
    def __init__(self, plot_config, pollutant_defs):
        self.config = plot_config
        self.pollutant_defs = pollutant_defs

    def create_2x2(self, data_file, pollutant, date):
        """Generate 2x2 panel plot."""
        # Load data
        ds = xr.open_dataset(data_file)

        # Get pollutant-specific settings
        p_config = self.pollutant_defs['pollutants'][pollutant]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure']['four_panel_size'],
                                dpi=self.config['figure']['dpi'])
        plt.subplots_adjust(wspace=self.config['layout']['2x2']['wspace'],
                           hspace=self.config['layout']['2x2']['hspace'])

        # Panel 0: Base
        ax = axes[0, 0]
        self._plot_panel(ax, ds['base'], pollutant, 'base',
                        self.config['layout']['2x2']['titles']['base'])

        # Panel 1: No Fire
        ax = axes[0, 1]
        self._plot_panel(ax, ds['nofire'], pollutant, 'nofire',
                        self.config['layout']['2x2']['titles']['nofire'])

        # Panel 2: Delta (absolute)
        ax = axes[1, 0]
        self._plot_panel(ax, ds['delta_abs'], pollutant, 'delta',
                        self.config['layout']['2x2']['titles']['delta_abs'])

        # Panel 3: Delta (percent)
        ax = axes[1, 1]
        self._plot_panel(ax, ds['delta_percent'], pollutant, 'delta_percent',
                        self.config['layout']['2x2']['titles']['delta_percent'])

        return fig, axes

    def _plot_panel(self, ax, data, pollutant, panel_type, title):
        """Plot a single panel."""
        # Get colormap and levels
        cmap = get_colormap(pollutant, panel_type, self.pollutant_defs)
        levels = get_levels(pollutant, panel_type, self.pollutant_defs)

        # Plot
        plot = data.plot(ax=ax, cmap=cmap, levels=levels, add_colorbar=True)

        # Add map features
        add_boundaries(ax, self.config)

        # Styling
        ax.set_title(title, fontsize=self.config['figure'].get('title_fontsize', 10))

        if self.config['map']['hide_axes']:
            ax.set_xlabel('')
            ax.set_ylabel('')

        if self.config['map']['hide_ticks']:
            ax.set_xticks([])
            ax.set_yticks([])

        return plot
```

#### `scripts/02_plotting/utils/ppt_builder.py`

```python
"""
PowerPoint presentation builder using python-pptx.
"""

from pptx import Presentation
from pptx.util import Inches
from pathlib import Path

class PowerPointBuilder:
    def __init__(self, output_path, config):
        self.output_path = Path(output_path)
        self.config = config
        self.prs = Presentation()

        # Add title slide
        self._add_title_slide()

    def _add_title_slide(self):
        """Add title slide."""
        title_slide_layout = self.prs.slide_layouts[0]
        slide = self.prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]

        title.text = "CMAQ Wildfire Analysis"
        subtitle.text = "Fire Impact Assessment - June 2023"

    def add_slide(self, title, image_path):
        """Add a slide with title and image."""
        blank_slide_layout = self.prs.slide_layouts[6]  # Blank
        slide = self.prs.slides.add_slide(blank_slide_layout)

        # Add title textbox
        left = Inches(0.5)
        top = Inches(0.5)
        width = Inches(9)
        height = Inches(0.8)
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.text = title

        # Add image
        img_config = self.config['powerpoint']['image_position']
        left = Inches(img_config['left'])
        top = Inches(img_config['top'])
        width = Inches(img_config['width'])

        slide.shapes.add_picture(str(image_path), left, top, width=width)

    def save(self):
        """Save presentation."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(self.output_path))
```

---

## 🚀 Master Orchestration Script

### `scripts/run_pipeline.py`

```python
"""
Master pipeline: Extract data → Generate plots → Create PowerPoint

Usage:
    python run_pipeline.py --date-range 2023-06-01 2023-06-30 --pollutants PM25 O3 CO
    python run_pipeline.py --quick-test  # Process June 1st only
"""

import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run full analysis pipeline')
    parser.add_argument('--date-range', nargs=2, required=True)
    parser.add_argument('--pollutants', nargs='+', default=['PM25', 'O3', 'CO'])
    parser.add_argument('--skip-analysis', action='store_true', help='Use existing processed data')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')

    args = parser.parse_args()

    # Step 1: Extract daily data
    if not args.skip_analysis:
        print("🔄 Step 1: Extracting daily data...")
        cmd = [
            'python', 'scripts/01_analysis/extract_daily_data.py',
            '--date-range', *args.date_range,
            '--pollutants', *args.pollutants
        ]
        if args.parallel:
            cmd.append('--parallel')
        subprocess.run(cmd, check=True)

    # Step 2: Generate plots
    if not args.skip_plots:
        print("🎨 Step 2: Generating plots...")
        cmd = [
            'python', 'scripts/02_plotting/generate_daily_plots.py',
            '--date-range', *args.date_range,
            '--pollutants', *args.pollutants,
            '--to-ppt'
        ]
        subprocess.run(cmd, check=True)

    print("✅ Pipeline complete!")

if __name__ == '__main__':
    main()
```

---

## 💾 Data Format Recommendations

### Processed Data Formats

1. **NetCDF (Primary)** - Gridded daily data
   - ✅ Preserves spatial structure
   - ✅ Includes metadata (projection, units, processing history)
   - ✅ Efficient for plotting
   - ✅ Standard in atmospheric science
   - File naming: `YYYY-MM-DD_POLLUTANT.nc`

2. **CSV (Secondary)** - Time series at city locations
   - ✅ Easy to share with collaborators
   - ✅ Good for statistical analysis in R/Python
   - ✅ Human-readable
   - File: `cities_timeseries_{POLLUTANT}.csv`

3. **Parquet (Optional)** - For very large tabular exports
   - ✅ 10x faster than CSV
   - ✅ Compressed (smaller files)
   - ✅ Preserves data types

4. **Excel (NOT Recommended)**
   - ❌ Slow for large datasets
   - ❌ Limited to 1 million rows
   - ❌ Can corrupt floating-point precision
   - ❌ Not reproducible

---

## 🎨 Plot Generation Details

### 2x2 Panel Structure

```
┌─────────────────────┬─────────────────────┐
│  Panel 1: BASE      │  Panel 2: NO FIRE   │
│  (Total with fires) │  (Background only)  │
│  - colormap: viridis│  - same colormap    │
│  - levels from      │  - adjusted levels  │
│    config           │                     │
└─────────────────────┴─────────────────────┘
┌─────────────────────┬─────────────────────┐
│  Panel 3: DELTA     │  Panel 4: DELTA %   │
│  (base - nofire)    │  (delta/base * 100) │
│  - colormap: YlOrRd │  - same colormap    │
│    or RdBu_r (O3)   │  - % levels         │
│  - units: μg/m³ ppb │  - units: %         │
└─────────────────────┴─────────────────────┘
```

### Pollutant-Specific Handling

- **PM2.5**: Always positive delta, YlOrRd colormap
- **O3**: Can be negative (NO titration in fresh smoke), RdBu_r diverging colormap
- **CO, Benzene, Toluene**: Positive delta, sequential colormap (Purples, YlOrRd)

---

## ⚙️ Key Features

✅ **Configuration-driven**: Change parameters without editing code
✅ **CLI interface**: Easy batch processing from command line
✅ **Parallel processing**: Process multiple days/pollutants simultaneously
✅ **Comprehensive logging**: Track what was processed, when, and any errors
✅ **Input validation**: Check data quality, missing files, invalid parameters
✅ **Metadata tracking**: Record processing parameters in output files
✅ **Progress bars**: Visual feedback for long operations (tqdm)
✅ **Error recovery**: Skip failed days, continue processing
✅ **Extensible**: Easy to add new pollutants, analysis methods, plot types
✅ **Reproducible**: Version control, documented workflows

---

## 📦 Dependencies

### Required Packages

```txt
# Core scientific stack
numpy>=1.22
pandas>=2.0
xarray>=2023.11
matplotlib>=3.7
scipy>=1.10

# CMAQ-specific tools
PseudoNetCDF>=3.2
pyrsig>=0.10
pycno>=0.3
cmaqsatproc>=0.2

# Plotting & export
cartopy>=0.22
python-pptx>=0.6.21
adjustText>=0.8  # For non-overlapping labels
plotly>=5.0  # For interactive plots

# Utilities
pyyaml>=6.0
tqdm>=4.65  # Progress bars
click>=8.0  # CLI framework (alternative to argparse)
loguru>=0.7  # Better logging

# Optional
pytest>=7.0  # For tests
black>=23.0  # Code formatting
```

### Installation

```bash
# Create conda environment
conda create -n cmaq-analysis python=3.10
conda activate cmaq-analysis

# Install dependencies
pip install -r requirements.txt

# Or use conda environment file
conda env create -f environment.yaml
```

---

## 🚀 Usage Examples

### Scenario 1: Process Single Day, One Pollutant

```bash
# Extract data
python scripts/01_analysis/extract_daily_data.py \
    --date 2023-06-07 \
    --pollutants PM25

# Generate plot
python scripts/02_plotting/generate_daily_plots.py \
    --date 2023-06-07 \
    --pollutants PM25 \
    --to-ppt --ppt-output figures/presentations/June7_PM25.pptx
```

### Scenario 2: Process Entire Month, All Pollutants

```bash
# Full pipeline (extraction + plotting)
python scripts/run_pipeline.py \
    --date-range 2023-06-01 2023-06-30 \
    --pollutants PM25 O3 CO BENZENE TOLUENE \
    --parallel
```

### Scenario 3: Re-plot Existing Data with New Styling

```bash
# Edit config/plot_config.yaml, then:
python scripts/02_plotting/generate_daily_plots.py \
    --date-range 2023-06-01 2023-06-30 \
    --pollutants all \
    --to-ppt
```

---

## 📊 PowerPoint Output Structure

```
Wildfire_Analysis_June2023.pptx
├── Slide 1: Title - "CMAQ Wildfire Analysis - June 2023"
├── Slide 2: PM2.5 - 2023-06-01 (2x2 image)
├── Slide 3: O3 - 2023-06-01 (2x2 image)
├── Slide 4: CO - 2023-06-01 (2x2 image)
├── Slide 5: PM2.5 - 2023-06-02 (2x2 image)
├── ...
└── Slide N: Summary (optional)
```

---

## ✅ Testing Strategy

### Unit Tests (`tests/`)

```python
# tests/test_calculator.py
def test_delta_calculation():
    """Test fire impact delta calculation"""
    base = xr.DataArray([10, 20, 30])
    nofire = xr.DataArray([5, 10, 15])
    calc = FireImpactCalculator(config, pollutant_defs)

    result = calc.calculate_delta(base, nofire)

    assert (result == xr.DataArray([5, 10, 15])).all()

def test_unit_conversion_ppb_to_ugm3():
    """Test CO ppb → μg/m³ conversion"""
    # Test with known values
    ...
```

### Integration Test

```bash
# Process one day as smoke test
python scripts/run_pipeline.py \
    --date-range 2023-06-01 2023-06-01 \
    --pollutants PM25 \
    --parallel
```

---

## 🔍 Advanced Features (Phase 2)

Once core functionality is solid, consider:

1. **Interactive Dashboards** (Plotly Dash / Streamlit)
   - Select date, pollutant, plot type interactively
   - Hover tooltips with values
   - Export custom plots

2. **Statistical Summaries**
   - Domain-wide statistics (mean, max, percentiles)
   - City-level time series CSV
   - Export to Excel with formatting

3. **Comparison Plots**
   - Side-by-side comparisons of different model versions
   - Difference plots (CRACMMv3 - CRACMMv2)

4. **Animation Generation**
   - Create MP4 videos of daily evolution
   - Use ffmpeg or matplotlib.animation

5. **Web API**
   - Flask/FastAPI server to serve processed data
   - Query API for specific dates/pollutants/locations

---

## 📋 Implementation Phases

### Phase 1: Foundation (Week 1)
1. ✅ Set up directory structure
2. ✅ Create configuration files (YAML)
3. ✅ Implement core utilities (config_loader, logger)
4. ✅ Build data_loader module
5. ✅ Build calculator module
6. ✅ Test with one day, one pollutant

### Phase 2: Analysis Pipeline (Week 2)
1. ✅ Implement exporter module (NetCDF + CSV)
2. ✅ Build extract_daily_data.py CLI
3. ✅ Add parallel processing
4. ✅ Add validation and error handling
5. ✅ Test with full June dataset

### Phase 3: Visualization (Week 3)
1. ✅ Implement plot_2x2 module
2. ✅ Build map_utils (boundaries, projections)
3. ✅ Implement colormap_registry
4. ✅ Build generate_daily_plots.py CLI
5. ✅ Test plot generation for all pollutants

### Phase 4: PowerPoint Export (Week 4)
1. ✅ Implement ppt_builder module
2. ✅ Integrate with generate_daily_plots.py
3. ✅ Add custom slide templates
4. ✅ Test full month → PPT workflow

### Phase 5: Integration & Testing
1. ✅ Build run_pipeline.py master script
2. ✅ Write unit tests
3. ✅ Write user documentation (README)
4. ✅ Optimize performance
5. ✅ Code review and refactoring

---

## 🎯 Discussion Points

### Questions to Address:

1. **Data format preference?**
   - NetCDF for gridded data? (recommended)
   - Also export CSV for city time series?

2. **Pollutants to support initially?**
   - Start with PM2.5, O3, CO?
   - Add Benzene, Toluene, photochemical age later?

3. **Date range?**
   - Just June 2023?
   - Or support any month/year?

4. **PowerPoint features?**
   - Simple image insertion?
   - Or add titles, annotations, data tables?

5. **Parallel processing?**
   - How many CPU cores available?
   - Process days in parallel? Pollutants?

6. **Additional outputs?**
   - Interactive HTML plots (Plotly)?
   - City time series CSV?
   - Domain statistics (mean, max, etc.)?

7. **Priority?**
   - Build analysis scripts first (Phase 1-2)?
   - Or plotting scripts first (Phase 3-4)?

---

## 📚 Documentation Plan

### README.md

- Installation instructions
- Quick start guide
- Usage examples
- Configuration guide
- Troubleshooting

### API Documentation

- Auto-generated from docstrings (Sphinx)
- Module/function reference
- Examples

### User Guide

- Step-by-step tutorials
- Common workflows
- Best practices

---

## ✨ Summary

This architecture provides:

✅ **Enterprise-grade structure** with clear separation of concerns
✅ **Configuration-driven** approach (no hard-coding)
✅ **Modular, testable, maintainable** codebase
✅ **Scalable** to large datasets and long time periods
✅ **Easy to extend** with new pollutants, analyses, visualizations
✅ **Comprehensive documentation** and testing
✅ **Production-ready** for operational use

**Estimated Development Time:** 3-4 weeks for full implementation

---

**Next Steps:** Review this plan, discuss modifications, and begin implementation!
