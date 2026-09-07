import argparse
import os
import time

import h5py
import numpy
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

QUANTILE_LOW = 0.158655253931
QUANTILE_HIGH = 0.841344746069
PHOTOMETRY_KEY = ['redshift_true', 'mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']


def redshift_bin(redshift, redshift_edge):
    '''
    Assign true redshifts to left-closed bins without duplicating edges
    '''
    index = numpy.searchsorted(redshift_edge, redshift, side='right') - 1
    last = len(redshift_edge) - 2
    index = numpy.where(redshift == redshift_edge[-1], last, index)
    select = (index >= 0) & (index <= last) & numpy.isfinite(redshift)
    return index, select


def measure_color(color, count):
    '''
    Measure the median and 1-sigma-equivalent percentile interval
    '''
    value = numpy.asarray(color, dtype=numpy.float64)
    value = value[numpy.isfinite(value)]
    number = len(value)
    
    median = numpy.nan
    quantile_low = numpy.nan
    quantile_high = numpy.nan
    plotted = number >= count
    
    if plotted:
        median = numpy.quantile(value, 0.5, method='linear')
        quantile_low, quantile_high = numpy.quantile(value, [QUANTILE_LOW, QUANTILE_HIGH], method='linear')
    
    return {
        'count': number,
        'median': median,
        'quantile_low': quantile_low,
        'quantile_high': quantile_high,
        'plotted': plotted
    }


def load_photometry(path):
    '''
    Load the photometry columns needed for the colour-redshift relation
    '''
    with h5py.File(path, 'r') as file:
        photometry = {key: numpy.asarray(file['photometry'][key][...], dtype=numpy.float64) for key in PHOTOMETRY_KEY}
    return photometry


def sample_color(photometry):
    '''
    Build the observed colours
    '''
    return [
        photometry['mag_u_lsst'] - photometry['mag_g_lsst'],
        photometry['mag_g_lsst'] - photometry['mag_r_lsst'],
        photometry['mag_i_lsst'] - photometry['mag_z_lsst'],
        photometry['mag_z_lsst'] - photometry['mag_y_lsst']
    ]


def main(tag, index, folder, count):
    '''
    Plot the colour-redshift relation before and after augmentation
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
        count (int): The minimum object count of a redshift grid
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print(f'Index: {index}')
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(figure_folder, f'{tag}/'), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, f'{tag}/RELATION/'), exist_ok=True)
    
    application_path = os.path.join(dataset_folder, f'{tag}/APPLICATION/DATA{index}.hdf5')
    degradation_path = os.path.join(dataset_folder, f'{tag}/DEGRADATION/DATA{index}.hdf5')
    combination_path = os.path.join(dataset_folder, f'{tag}/COMBINATION/DATA{index}.hdf5')
    
    # Load
    application_photometry = load_photometry(application_path)
    degradation_photometry = load_photometry(degradation_path)
    combination_photometry = load_photometry(combination_path)
    
    sample_name = ['Application', 'Degradation', 'Combination']
    sample_photometry = [application_photometry, degradation_photometry, combination_photometry]
    sample_style = {
        'Application': {'color': 'black', 'marker': 'o', 'markersize': 6.5, 'offset': 0.0},
        'Degradation': {'color': 'darkgreen', 'marker': 's', 'markersize': 6.0, 'offset': -0.05},
        'Combination': {'color': 'darkorange', 'marker': '^', 'markersize': 7.0, 'offset': +0.05}
    }
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2026/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Redshift
    redshift_z1 = 0.0
    redshift_z2 = 3.0
    redshift_width = 0.3
    redshift_size = int(numpy.round((redshift_z2 - redshift_z1) / redshift_width))
    redshift_edge = numpy.linspace(redshift_z1, redshift_z2, redshift_size + 1)
    redshift_center = 0.5 * (redshift_edge[:-1] + redshift_edge[1:])
    
    # Magnitude
    magnitude_lower = 16.0
    magnitude_upper = 25.3
    magnitude_edge1 = 21.0
    magnitude_edge2 = 22.5
    magnitude_edge3 = 24.0
    magnitude_edge = [magnitude_lower, magnitude_edge1, magnitude_edge2, magnitude_edge3, magnitude_upper]
    
    # Color
    label_list = [r'$u - g$', r'$g - r$', r'$i - z$', r'$z - y$']
    
    sample_color_list = [sample_color(photometry) for photometry in sample_photometry]
    sample_redshift = [photometry['redshift_true'] for photometry in sample_photometry]
    sample_magnitude = [photometry['mag_i_lsst'] for photometry in sample_photometry]
    
    # Statistics
    measurement = {}
    
    for n, name in enumerate(sample_name):
        redshift_index, redshift_select = redshift_bin(sample_redshift[n], redshift_edge)
        measurement[name] = {}
        
        for i in range(len(magnitude_edge) - 1):
            magnitude_select = (magnitude_edge[i] < sample_magnitude[n]) & (sample_magnitude[n] < magnitude_edge[i + 1])
            measurement[name][i] = {}
            
            for j in range(len(label_list)):
                color_array = sample_color_list[n][j]
                statistic_list = []
                
                for k in range(redshift_size):
                    select = redshift_select & magnitude_select & (redshift_index == k) & numpy.isfinite(color_array)
                    statistic_list.append(measure_color(color_array[select], count))
                
                measurement[name][i][j] = statistic_list
    
    color1 = []
    color2 = []
    for j in range(len(label_list)):
        endpoint = []
        for name in sample_name:
            for i in range(len(magnitude_edge) - 1):
                for statistic in measurement[name][i][j]:
                    if statistic['plotted']:
                        endpoint.append(statistic['quantile_low'])
                        endpoint.append(statistic['quantile_high'])
        endpoint = numpy.asarray(endpoint, dtype=numpy.float64)
        endpoint = endpoint[numpy.isfinite(endpoint)]
        lower = numpy.min(endpoint)
        upper = numpy.max(endpoint)
        pad = max(0.05, 0.08 * (upper - lower))
        color1.append(float(numpy.floor((lower - pad) * 10.0) / 10.0))
        color2.append(float(numpy.ceil((upper + pad) * 10.0) / 10.0))
    
    # Figure
    figure = pyplot.figure(figsize=(20, 20))
    gridspec = GridSpec(nrows=len(label_list), ncols=len(magnitude_edge) - 1, figure=figure, hspace=0.0, wspace=0.0, top=0.94)
    
    legend_handle = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', markersize=6.5, linewidth=1.8, label=r'$\mathtt{Application}$'),
        Patch(facecolor='0.82', edgecolor='none', label=r'$\mathtt{Application}$ central $68.27\%$ interval'),
        Line2D([0], [0], color='darkgreen', marker='s', linestyle='-', markersize=6.0, linewidth=1.4, label=r'$\mathtt{Degradation}$'),
        Line2D([0], [0], color='darkorange', marker='^', linestyle='-', markersize=7.0, linewidth=1.4, label=r'$\mathtt{Combination}$')
    ]
    
    for i in range(len(magnitude_edge) - 1):
        for j in range(len(label_list)):
            plot = figure.add_subplot(gridspec[j, i])
            
            application = measurement['Application'][i][j]
            application_median = numpy.array([statistic['median'] if statistic['plotted'] else numpy.nan for statistic in application], dtype=numpy.float64)
            application_low = numpy.array([statistic['quantile_low'] if statistic['plotted'] else numpy.nan for statistic in application], dtype=numpy.float64)
            application_high = numpy.array([statistic['quantile_high'] if statistic['plotted'] else numpy.nan for statistic in application], dtype=numpy.float64)
            application_band = numpy.isfinite(application_low) & numpy.isfinite(application_high)
            
            plot.fill_between(
                redshift_center,
                application_low,
                application_high,
                where=application_band,
                interpolate=False,
                color='0.82',
                linewidth=0.0,
                zorder=1
            )
            plot.plot(redshift_center, application_median, color='black', marker='o', linestyle='-', markersize=6.5, linewidth=1.8, zorder=4)
            
            for name in ['Degradation', 'Combination']:
                style = sample_style[name]
                statistic_list = measurement[name][i][j]
                median = numpy.array([statistic['median'] if statistic['plotted'] else numpy.nan for statistic in statistic_list], dtype=numpy.float64)
                quantile_low = numpy.array([statistic['quantile_low'] if statistic['plotted'] else numpy.nan for statistic in statistic_list], dtype=numpy.float64)
                quantile_high = numpy.array([statistic['quantile_high'] if statistic['plotted'] else numpy.nan for statistic in statistic_list], dtype=numpy.float64)
                yerr = numpy.vstack([median - quantile_low, quantile_high - median])
                
                plot.errorbar(
                    redshift_center + style['offset'],
                    median,
                    yerr=yerr,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle='-',
                    markersize=style['markersize'],
                    linewidth=1.4,
                    elinewidth=1.0,
                    capsize=2.5,
                    capthick=1.0,
                    zorder=3
                )
            
            plot.set_xticks([0.0, 0.6, 1.2, 1.8, 2.4])
            plot.set_xlim(redshift_z1, redshift_z2)
            plot.set_ylim(color1[j], color2[j])
            
            if i == 0:
                plot.set_ylabel(label_list[j])
            else:
                plot.set_yticklabels([])
            
            if j == len(label_list) - 1:
                plot.set_xlabel(r'$z_\mathrm{true}$')
            else:
                plot.set_xticklabels([])
            
            if j == 0:
                plot.set_title(rf'${magnitude_edge[i]:.1f} < i < {magnitude_edge[i + 1]:.1f}$')
    
    figure.legend(handles=legend_handle, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=24)
    
    # Save
    figure.savefig(os.path.join(figure_folder, f'{tag}/RELATION/FIGURE{index}.pdf'), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print(f'Time: {duration:.2f} minutes')
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Relation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    PARSE.add_argument('--count', type=int, required=True, help='The minimum object count of a redshift grid')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    COUNT = PARSE.parse_args().count
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER, COUNT)
