import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot, gridspec


def main(tag, index, folder):
    '''
    Plot the figures of the redshift estimation
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the plotter
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    comparison_folder = os.path.join(folder, 'COMPARISON/')
    
    # Redshift
    z1_average_lens = 0.0
    z2_average_lens = 1.6
    average_lens_size = 8
    z_average_lens = numpy.linspace(z1_average_lens, z2_average_lens, average_lens_size + 1)
    z_bin_lens = (z_average_lens[1:] + z_average_lens[:-1]) / 2
    
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_source_size = 10
    z_average_source = numpy.linspace(z1_average_source, z2_average_source, average_source_size + 1)
    z_bin_source = (z_average_source[1:] + z_average_source[:-1]) / 2
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        
        delta_lens = file['delta_lens'][...]
        delta_source = file['delta_source'][...]
        
        sigma_lens = file['sigma_lens'][...]
        sigma_source = file['sigma_source'][...]
        
        rate_lens = file['rate_lens'][...]
        rate_source = file['rate_source'][...]
        
        fraction_lens = file['fraction_lens'][...]
        fraction_source = file['fraction_source'][...]
    
    # Comparison
    with h5py.File(os.path.join(comparison_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        comparison_sigma_lens = file['sigma_lens'][...]
        comparison_sigma_source = file['sigma_source'][...]
        
        comparison_delta_lens = file['delta_lens'][...]
        comparison_delta_source = file['delta_source'][...]
        
        comparison_rate_lens = file['rate_lens'][...]
        comparison_rate_source = file['rate_source'][...]
        
        comparison_fraction_lens = file['fraction_lens'][...]
        comparison_fraction_source = file['fraction_source'][...]
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    figure = pyplot.figure(figsize=(16, 16))
    plot_list = gridspec.GridSpec(nrows=4, ncols=2, figure=figure)
    
    # Plot lens delta 
    plot = figure.add_subplot(plot_list[0, 0])
    
    plot.plot(z_bin_lens, delta_lens, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_lens, comparison_delta_lens, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_lens, delta_lens, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_lens, comparison_delta_lens, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.005, 0.015)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathtt{lens}$')
    plot.set_ylabel(r'$\bar{\delta}_z$')
    
    # Plot lens sigma 
    plot = figure.add_subplot(plot_list[1, 0])
    
    plot.plot(z_bin_lens, sigma_lens, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_lens, comparison_sigma_lens, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$') 
    
    plot.scatter(z_bin_lens, sigma_lens, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_lens, comparison_sigma_lens, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.005, 0.015)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$')
    
    # Plot lens fraction
    plot = figure.add_subplot(plot_list[2, 0])
    
    plot.plot(z_bin_lens, fraction_lens, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_lens, comparison_fraction_lens, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_lens, fraction_lens, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_lens, comparison_fraction_lens, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.005, 0.015)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$')
    
    # Plot lens rate
    plot = figure.add_subplot(plot_list[3, 0])
    
    plot.plot(z_bin_lens, rate_lens, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_lens, comparison_rate_lens, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_lens, rate_lens, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_lens, comparison_rate_lens, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.005, 0.015)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_ylabel(r'$r_\mathrm{c}$')
    plot.set_xlabel(r'$z_\mathrm{true}$')
    
    # Plot source delta 
    plot = figure.add_subplot(plot_list[0, 1])
    
    plot.plot(z_bin_source, delta_source, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_source, comparison_delta_source, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')    
    
    plot.scatter(z_bin_source, delta_source, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_source, comparison_delta_source, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.100, 0.800)
    plot.legend(loc='upper left', fontsize=25)
    plot.set_xlim(z1_average_source, z2_average_source) 
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathtt{source}$') 
    
    # Plot source sigma 
    plot = figure.add_subplot(plot_list[1, 1])
    
    plot.plot(z_bin_source, sigma_source, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_source, comparison_sigma_source, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_source, sigma_source, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_source, comparison_sigma_source, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_xticklabels([])
    plot.set_ylim(-0.100, 0.400)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    # Plot source fraction
    plot = figure.add_subplot(plot_list[2, 1])
    
    plot.plot(z_bin_source, fraction_source, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_source, comparison_fraction_source, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_source, fraction_source, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_source, comparison_fraction_source, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_xticklabels([])
    plot.set_ylim(-0.100, 1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    # Plot source rate
    plot = figure.add_subplot(plot_list[3, 1])
    
    plot.plot(z_bin_source, rate_source, color='darkred', linestyle='-', linewidth=2.5, label=r'$\mathrm{No \, Augmentation}$')
    
    plot.plot(z_bin_source, comparison_rate_source, color='darkblue', linestyle='-', linewidth=2.5, label=r'$\mathrm{With \, Augmentation}$')
    
    plot.scatter(z_bin_source, rate_source, marker='s', s=100, alpha=0.8, facecolors='none', edgecolors='darkred', linewidths=2.5)
    
    plot.scatter(z_bin_source, comparison_rate_source, marker='d', s=100, alpha=0.8, facecolors='none', edgecolors='darkblue', linewidths=2.5)
    
    plot.set_ylim(-0.100, 1.000)
    plot.set_xlabel(r'$z_\mathrm{true}$')
    plot.set_xlim(z1_average_source, z2_average_source)
    
    # Save
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/METRIC/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.2, hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/METRIC/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Metric')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)