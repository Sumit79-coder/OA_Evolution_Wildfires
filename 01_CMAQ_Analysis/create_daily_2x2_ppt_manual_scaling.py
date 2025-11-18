"""
CMAQ Daily 2x2 Fire Impact Maps - PowerPoint Generator
MANUAL SCALING VERSION

This script creates daily 2x2 comparison maps showing fire impacts on air pollutants.
You manually define the color scale levels for each pollutant.

Author: Generated for CMAQ Fire Analysis
Date: 2025-01-18
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyrsig
import pycno
from datetime import datetime, timedelta
from pptx import Presentation
from pptx.util import Inches
import warnings
warnings.filterwarnings('ignore')

# ============= USER CONFIGURATION =============
# Pollutant to analyze
POLLUTANT = 'O3'  # Options: 'O3', 'PM25_TOT', 'CO', 'BENZENE', 'TOLUENE', 'PHENOL'

# Unit conversion
CONVERT_TO_UGM3 = False  # True: convert to μg/m³, False: keep native units (ppb for gases)

# Paths
BASE_DIR = r'D:/Raw_Data/CMAQ_Model/'
OUTPUT_DIR = r'C:\Users\smtku\OA_Evolution_Wildfires\01_CMAQ_Analysis\figures\daily_2x2_maps'

# Daily aggregation method
DAILY_METHOD = 'mean'  # Options: 'mean' or 'max'

# Start date (first day of data)
START_DATE = datetime(2023, 6, 1)
# ==============================================


# ============= POLLUTANT CONFIGURATION =============
POLLUTANT_CONFIG = {
    'O3': {
        'var_name': 'O3',
        'display_name': 'O₃',
        'native_units': 'ppb',
        'colormap_base': 'viridis',
        'colormap_delta': 'RdBu_r',
        'levels_base': [0, 10, 20, 30, 40, 50, 60, 70, 80],
        'levels_delta': [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20],
        'levels_percent': [-50, -20, -10, -5, 0, 5, 10, 20, 50, 100],
        'can_be_negative': True,
        'molecular_weight': 48.00,
    },
    'PM25_TOT': {
        'var_name': 'PM25_TOT',
        'display_name': 'PM₂.₅',
        'native_units': 'μg/m³',
        'colormap_base': 'viridis',
        'colormap_delta': 'YlOrRd',
        'levels_base': [0, 5, 10, 15, 20, 30, 40, 50, 60, 80],
        'levels_delta': [-0.5, 0, 0.5, 1, 2, 5, 10, 20, 50, 100],
        'levels_percent': [-1, 0, 5, 10, 20, 30, 50, 100, 200, 500],
        'can_be_negative': False,
        'molecular_weight': None,  # Already in μg/m³
    },
    'CO': {
        'var_name': 'CO',
        'display_name': 'CO',
        'native_units': 'ppb',
        'colormap_base': 'viridis',
        'colormap_delta': 'YlOrRd',
        'levels_base': [0, 100, 200, 300, 400, 500, 750, 1000, 2000],
        'levels_delta': [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        'levels_percent': [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        'can_be_negative': False,
        'molecular_weight': 28.01,
    },
    'BENZENE': {
        'var_name': 'BENZENE',
        'display_name': 'Benzene',
        'native_units': 'ppb',
        'colormap_base': 'viridis',
        'colormap_delta': 'YlOrRd',
        'levels_base': [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        'levels_delta': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'levels_percent': [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        'can_be_negative': False,
        'molecular_weight': 78.11,
    },
    'TOLUENE': {
        'var_name': 'TOLUENE',
        'display_name': 'Toluene',
        'native_units': 'ppb',
        'colormap_base': 'viridis',
        'colormap_delta': 'YlOrRd',
        'levels_base': [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        'levels_delta': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'levels_percent': [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        'can_be_negative': False,
        'molecular_weight': 92.14,
    },
    'PHENOL': {
        'var_name': 'PHENOL',
        'display_name': 'Phenol',
        'native_units': 'ppb',
        'colormap_base': 'viridis',
        'colormap_delta': 'YlOrRd',
        'levels_base': [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'levels_delta': [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2],
        'levels_percent': [0, 10, 20, 50, 100, 200, 500, 1000, 2000],
        'can_be_negative': False,
        'molecular_weight': 94.11,
    },
}
# ===================================================


def load_cmaq_data():
    """Load CMAQ base and no-fire data"""
    print("Loading CMAQ data...")

    cmaq_base_path = BASE_DIR + 'netcdffiles/COMBINE_ACONC_cmaq6acracmm3_base_2023_12US4_202306.nc'
    cmaq_nofire_path = BASE_DIR + 'netcdffiles/COMBINE_ACONC_cmaq6acracmm3_nofire_2023_12US4_202306.nc'
    metcro2d_path = BASE_DIR + 'MOD3DATA_MET/METCRO2D.12US4.35L.230701'
    gridcro2d_path = BASE_DIR + 'MOD3DATA_MET/GRIDCRO2D.12US4.35L.230701'

    baseconc = pyrsig.open_ioapi(cmaq_base_path)
    nofireconc = pyrsig.open_ioapi(cmaq_nofire_path)
    metcro2d = pyrsig.open_ioapi(metcro2d_path)

    # Load projection info
    cno = pycno.cno(baseconc.crs_proj4)

    print("Data loaded successfully!")
    return baseconc, nofireconc, metcro2d, cno


def calculate_fire_impact(baseconc, nofireconc, metcro2d, config):
    """Calculate fire impact (base - nofire) for the pollutant"""
    print(f"Calculating fire impact for {config['display_name']}...")

    var_name = config['var_name']

    # Get base and nofire concentrations
    base = baseconc[var_name]
    nofire = nofireconc[var_name]

    # Select surface layer (LAY=0) if data has layer dimension
    if len(base.shape) == 4:  # (TSTEP, LAY, ROW, COL)
        print("  Selecting surface layer (LAY=0)...")
        base = base[:, 0, :, :]
        nofire = nofire[:, 0, :, :]

    # Convert to μg/m³ if requested and if it's a gas (ppb)
    if CONVERT_TO_UGM3 and config['native_units'] == 'ppb':
        print(f"Converting from {config['native_units']} to μg/m³...")
        air_dens = metcro2d['DENS']  # Air density (kg/m³)

        # Select surface layer for air density if needed
        if len(air_dens.shape) == 4:
            air_dens = air_dens[:, 0, :, :]

        mw = config['molecular_weight']

        # Convert: μg/m³ = ppb × air_dens × MW / 28.9628
        base = base * air_dens * mw / 28.9628
        nofire = nofire * air_dens * mw / 28.9628

        units = 'μg/m³'
    else:
        units = config['native_units']

    # Calculate delta
    delta = base - nofire

    print(f"Fire impact calculated! Units: {units}")
    return base, nofire, delta, units


def aggregate_to_daily(data, method='mean'):
    """Aggregate hourly data to daily"""
    print(f"Aggregating to daily {method}...")

    # Reshape from (720, ROW, COL) to (30, 24, ROW, COL)
    n_days = 30
    n_hours = 24

    # Get data array
    data_array = data.values

    # Reshape
    shape = (n_days, n_hours) + data_array.shape[1:]
    data_reshaped = data_array.reshape(shape)

    # Aggregate
    if method == 'mean':
        daily_data = np.mean(data_reshaped, axis=1)
    elif method == 'max':
        daily_data = np.max(data_reshaped, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Daily aggregation complete! Shape: {daily_data.shape}")
    return daily_data


def set_map(ax, cno, title, colorlabel):
    """Format map plots"""
    ax.colorbar.set_label(colorlabel, rotation=270, labelpad=15, fontsize=9)
    ax.axes.set_xlabel('')
    ax.axes.set_xticklabels('')
    ax.axes.set_ylabel('')
    ax.axes.set_yticklabels('')
    plt.setp(ax.axes, **dict(title=title))
    cno.drawstates(ax=ax.axes, linewidth=0.2)


def create_daily_plot(base_day, nofire_day, delta_day, config, units, date_str, cno, day_num):
    """Create 2x2 plot for a single day"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.35, top=0.92, bottom=0.08, left=0.05, right=0.95)

    # Overall title
    fig.suptitle(f'Wildfire Impact on {config["display_name"]} - {date_str}\n(CMAQv6a1 CRACMM3, 12US4 Domain)',
                 fontsize=14, fontweight='bold')

    # Panel (a): Base simulation
    plt.sca(axes[0, 0])
    pv = axes[0, 0].contourf(base_day, cmap=config['colormap_base'],
                              levels=config['levels_base'], extend='both')
    title_a = f'(a) Base Simulation\nMean: {np.nanmean(base_day):.2f}, Max: {np.nanmax(base_day):.1f} {units}'
    cb = plt.colorbar(pv, ax=axes[0, 0])
    set_map(cb, cno, title_a, f'{config["display_name"]} ({units})')

    # Panel (b): No-fire simulation
    plt.sca(axes[0, 1])
    pv = axes[0, 1].contourf(nofire_day, cmap=config['colormap_base'],
                              levels=config['levels_base'], extend='both')
    title_b = f'(b) No-Fire Scenario\nMean: {np.nanmean(nofire_day):.2f}, Max: {np.nanmax(nofire_day):.1f} {units}'
    cb = plt.colorbar(pv, ax=axes[0, 1])
    set_map(cb, cno, title_b, f'{config["display_name"]} ({units})')

    # Panel (c): Absolute delta
    plt.sca(axes[1, 0])
    pv = axes[1, 0].contourf(delta_day, cmap=config['colormap_delta'],
                              levels=config['levels_delta'], extend='both')
    title_c = f'(c) Fire Impact: Δ{config["display_name"]} (Base − No Fire)\nMean: {np.nanmean(delta_day):.2f}, Range: [{np.nanmin(delta_day):.2f}, {np.nanmax(delta_day):.2f}] {units}'
    cb = plt.colorbar(pv, ax=axes[1, 0])
    set_map(cb, cno, title_c, f'Δ{config["display_name"]} ({units})')

    # Panel (d): Percent delta
    percent_delta = ((base_day - nofire_day) / nofire_day * 100)
    percent_delta = np.where(np.isinf(percent_delta), np.nan, percent_delta)  # Remove infinities

    plt.sca(axes[1, 1])
    pv = axes[1, 1].contourf(percent_delta, cmap=config['colormap_delta'],
                              levels=config['levels_percent'], extend='both')
    title_d = f'(d) Relative Fire Impact: %Δ{config["display_name"]}\nMean: {np.nanmean(percent_delta):.1f}%, Range: [{np.nanmin(percent_delta):.0f}%, {np.nanmax(percent_delta):.0f}%]'
    cb = plt.colorbar(pv, ax=axes[1, 1])
    set_map(cb, cno, title_d, '% Change')

    # Save to temporary file
    temp_file = f'temp_day_{day_num:02d}.png'
    plt.savefig(temp_file, dpi=150, bbox_inches='tight')
    plt.close()

    return temp_file


def create_powerpoint(image_files, dates, output_path):
    """Create PowerPoint presentation with daily maps"""
    print("Creating PowerPoint presentation...")

    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Widescreen (16:9)
    prs.slide_height = Inches(7.5)

    for img_file, date_str in zip(image_files, dates):
        # Add blank slide
        blank_slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(blank_slide_layout)

        # Add image (full slide)
        left = Inches(0.5)
        top = Inches(0.5)
        width = Inches(12.333)
        slide.shapes.add_picture(img_file, left, top, width=width)

    # Save presentation
    prs.save(output_path)
    print(f"PowerPoint saved: {output_path}")

    # Clean up temporary files
    for img_file in image_files:
        if os.path.exists(img_file):
            os.remove(img_file)
    print("Temporary files cleaned up.")


def main():
    """Main execution function"""
    print("="*60)
    print("CMAQ Daily 2x2 Fire Impact Maps Generator")
    print("Manual Scaling Version")
    print("="*60)

    # Validate pollutant
    if POLLUTANT not in POLLUTANT_CONFIG:
        raise ValueError(f"Unknown pollutant: {POLLUTANT}. Options: {list(POLLUTANT_CONFIG.keys())}")

    config = POLLUTANT_CONFIG[POLLUTANT]
    print(f"\nPollutant: {config['display_name']}")
    print(f"Native units: {config['native_units']}")
    print(f"Convert to μg/m³: {CONVERT_TO_UGM3}")
    print(f"Daily aggregation: {DAILY_METHOD}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    baseconc, nofireconc, metcro2d, cno = load_cmaq_data()

    # Calculate fire impact
    base, nofire, delta, units = calculate_fire_impact(baseconc, nofireconc, metcro2d, config)

    # Aggregate to daily
    base_daily = aggregate_to_daily(base, method=DAILY_METHOD)
    nofire_daily = aggregate_to_daily(nofire, method=DAILY_METHOD)
    delta_daily = aggregate_to_daily(delta, method=DAILY_METHOD)

    # Generate plots for each day
    print("\nGenerating daily plots...")
    image_files = []
    dates = []

    for day in range(30):
        date = START_DATE + timedelta(days=day)
        date_str = date.strftime('%Y-%m-%d')
        dates.append(date_str)

        print(f"  Processing {date_str} (Day {day+1}/30)...")

        img_file = create_daily_plot(
            base_daily[day],
            nofire_daily[day],
            delta_daily[day],
            config,
            units,
            date_str,
            cno,
            day + 1
        )
        image_files.append(img_file)

    # Create output filename
    unit_str = units.replace('/', 'per').replace('³', '3').replace('μg', 'ug')
    output_filename = f"{config['var_name']}_{unit_str}_daily_2x2_June2023.pptx"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Create PowerPoint
    create_powerpoint(image_files, dates, output_path)

    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
