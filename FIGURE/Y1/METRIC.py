import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot, gridspec


def main(tag, number, folder):
    '''
    Plot the figures of the redshift estimation
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the plotter
    '''
    # Start
    start = time.time()
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    comparison_folder = os.path.join(folder, 'COMPARISON/')
    
    # Redshift
    z1_average_lens = 0.0
    z2_average_lens = 1.5
    average_size_lens = 5
    average_delta_lens = (z2_average_lens - z1_average_lens) / average_size_lens
    z_average_lens = numpy.linspace(z1_average_lens + average_delta_lens / 2, z2_average_lens - average_delta_lens / 2, average_size_lens)
    
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_size_source = 10
    average_delta_source = (z2_average_source - z1_average_source) / average_size_source
    z_average_source = numpy.linspace(z1_average_source + average_delta_source / 2, z2_average_source - average_delta_source / 2, average_size_source)
    
    # Select
    delta_lens = numpy.zeros((number + 1, average_size_lens))
    delta_source = numpy.zeros((number + 1, average_size_source))
    
    sigma_lens = numpy.zeros((number + 1, average_size_lens))
    sigma_source = numpy.zeros((number + 1, average_size_source))
    
    fraction_lens = numpy.zeros((number + 1, average_size_lens))
    fraction_source = numpy.zeros((number + 1, average_size_source))
    
    rate_lens = numpy.zeros((number + 1, average_size_lens))
    rate_source = numpy.zeros((number + 1, average_size_source))
    
    divergence_lens = numpy.zeros((number + 1, average_size_lens))
    divergence_source = numpy.zeros((number + 1, average_size_source))
    
    # Loop
    for index in range(number + 1):
        with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            
            delta_lens[index, :] = file['delta_lens'][...]
            delta_source[index, :] = file['delta_source'][...]
            
            sigma_lens[index, :] = file['sigma_lens'][...]
            sigma_source[index, :] = file['sigma_source'][...]
            
            fraction_lens[index, :] = file['fraction_lens'][...]
            fraction_source[index, :] = file['fraction_source'][...]
            
            rate_lens[index, :] = file['rate_lens'][...]
            rate_source[index, :] = file['rate_source'][...]
            
            divergence_lens[index, :] = file['divergence_lens'][...]
            divergence_source[index, :] = file['divergence_source'][...]
    
    # Reference
    reference_delta_lens = numpy.zeros((number + 1, average_size_lens))
    reference_delta_source = numpy.zeros((number + 1, average_size_source))
    
    reference_sigma_lens = numpy.zeros((number + 1, average_size_lens))
    reference_sigma_source = numpy.zeros((number + 1, average_size_source))
    
    reference_fraction_lens = numpy.zeros((number + 1, average_size_lens))
    reference_fraction_source = numpy.zeros((number + 1, average_size_source))
    
    reference_rate_lens = numpy.zeros((number + 1, average_size_lens))
    reference_rate_source = numpy.zeros((number + 1, average_size_source))
    
    reference_divergence_lens = numpy.zeros((number + 1, average_size_lens))
    reference_divergence_source = numpy.zeros((number + 1, average_size_source))
    
    # Loop
    for index in range(number + 1):
        with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            reference_delta_lens[index, :] = file['delta_lens'][...]
            reference_delta_source[index, :] = file['delta_source'][...]
            
            reference_sigma_lens[index, :] = file['sigma_lens'][...]
            reference_sigma_source[index, :] = file['sigma_source'][...]
            
            reference_fraction_lens[index, :] = file['fraction_lens'][...]
            reference_fraction_source[index, :] = file['fraction_source'][...]
            
            reference_rate_lens[index, :] = file['rate_lens'][...]
            reference_rate_source[index, :] = file['rate_source'][...]
            
            reference_divergence_lens[index, :] = file['divergence_lens'][...]
            reference_divergence_source[index, :] = file['divergence_source'][...]
    
    # Comparison
    comparison_delta_lens = numpy.zeros((number + 1, average_size_lens))
    comparison_delta_source = numpy.zeros((number + 1, average_size_source))
    
    comparison_sigma_lens = numpy.zeros((number + 1, average_size_lens))
    comparison_sigma_source = numpy.zeros((number + 1, average_size_source))
    
    comparison_fraction_lens = numpy.zeros((number + 1, average_size_lens))
    comparison_fraction_source = numpy.zeros((number + 1, average_size_source))
    
    comparison_rate_lens = numpy.zeros((number + 1, average_size_lens))
    comparison_rate_source = numpy.zeros((number + 1, average_size_source))
    
    comparison_divergence_lens = numpy.zeros((number + 1, average_size_lens))
    comparison_divergence_source = numpy.zeros((number + 1, average_size_source))
    
    # Loop
    for index in range(number + 1):
        with h5py.File(os.path.join(comparison_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            comparison_delta_lens[index, :] = file['delta_lens'][...]
            comparison_delta_source[index, :] = file['delta_source'][...]
            
            comparison_sigma_lens[index, :] = file['sigma_lens'][...]
            comparison_sigma_source[index, :] = file['sigma_source'][...]
            
            comparison_fraction_lens[index, :] = file['fraction_lens'][...]
            comparison_fraction_source[index, :] = file['fraction_source'][...]
            
            comparison_rate_lens[index, :] = file['rate_lens'][...]
            comparison_rate_source[index, :] = file['rate_source'][...]
            
            comparison_divergence_lens[index, :] = file['divergence_lens'][...]
            comparison_divergence_source[index, :] = file['divergence_source'][...]
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    figure = pyplot.figure(figsize=(16, 20))
    plot_list = gridspec.GridSpec(nrows=5, ncols=2, figure=figure)
    
    # Plot lens delta 
    plot = figure.add_subplot(plot_list[0, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(delta_lens, axis=0), yerr=numpy.std(delta_lens, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(reference_delta_lens, axis=0), yerr=numpy.std(reference_delta_lens, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(comparison_delta_lens, axis=0), yerr=numpy.std(comparison_delta_lens, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.006, +0.060)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{lens}$')
    plot.set_ylabel(r'$\tilde{\delta}_z$')
    
    # Plot lens sigma 
    plot = figure.add_subplot(plot_list[1, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(sigma_lens, axis=0), yerr=numpy.std(sigma_lens, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(reference_sigma_lens, axis=0), yerr=numpy.std(reference_sigma_lens, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(comparison_sigma_lens, axis=0), yerr=numpy.std(comparison_sigma_lens, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.006, +0.060)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$')
    
    # Plot lens fraction
    plot = figure.add_subplot(plot_list[2, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(fraction_lens, axis=0), yerr=numpy.std(fraction_lens, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(reference_fraction_lens, axis=0), yerr=numpy.std(reference_fraction_lens, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(comparison_fraction_lens, axis=0), yerr=numpy.std(comparison_fraction_lens, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$')
    
    # Plot lens rate
    plot = figure.add_subplot(plot_list[3, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(rate_lens, axis=0), yerr=numpy.std(rate_lens, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(reference_rate_lens, axis=0), yerr=numpy.std(reference_rate_lens, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(comparison_rate_lens, axis=0), yerr=numpy.std(comparison_rate_lens, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$')
    
    # Plot lens divergence
    plot = figure.add_subplot(plot_list[4, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(divergence_lens, axis=0), yerr=numpy.std(divergence_lens, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(reference_divergence_lens, axis=0), yerr=numpy.std(reference_divergence_lens, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=numpy.mean(comparison_divergence_lens, axis=0), yerr=numpy.std(comparison_divergence_lens, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.500, +5.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\mathcal{D}_q$')
    plot.set_xlabel(r'$z_\mathrm{true}$')
    
    # Plot source delta 
    plot = figure.add_subplot(plot_list[0, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.mean(delta_source, axis=0), yerr=numpy.std(delta_source, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(reference_delta_source, axis=0), yerr=numpy.std(reference_delta_source, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(comparison_delta_source, axis=0), yerr=numpy.std(comparison_delta_source, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{source}$')
    plot.set_ylabel(r'$\tilde{\delta}_z$')
    
    # Plot source sigma 
    plot = figure.add_subplot(plot_list[1, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.mean(sigma_source, axis=0), yerr=numpy.std(sigma_source, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(reference_sigma_source, axis=0), yerr=numpy.std(reference_sigma_source, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(comparison_sigma_source, axis=0), yerr=numpy.std(comparison_sigma_source, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.050, +0.500)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$')
    
    # Plot source fraction
    plot = figure.add_subplot(plot_list[2, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.mean(fraction_source, axis=0), yerr=numpy.std(fraction_source, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(reference_fraction_source, axis=0), yerr=numpy.std(reference_fraction_source, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(comparison_fraction_source, axis=0), yerr=numpy.std(comparison_fraction_source, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$')
    
    # Plot source rate
    plot = figure.add_subplot(plot_list[3, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.mean(rate_source, axis=0), yerr=numpy.std(rate_source, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(reference_rate_source, axis=0), yerr=numpy.std(reference_rate_source, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(comparison_rate_source, axis=0), yerr=numpy.std(comparison_rate_source, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$')
    
    # Plot source divergence
    plot = figure.add_subplot(plot_list[4, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.mean(divergence_source, axis=0), yerr=numpy.std(divergence_source, axis=0), color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(reference_divergence_source, axis=0), yerr=numpy.std(reference_divergence_source, axis=0), color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=numpy.mean(comparison_divergence_source, axis=0), yerr=numpy.std(comparison_divergence_source, axis=0), color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8)
    
    plot.set_ylim(-0.500, +5.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\mathcal{D}_q$')
    plot.set_xlabel(r'$z_\mathrm{true}$')
    
    # Save
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/METRIC/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(wspace=0.2, hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/METRIC/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight')
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
    PARSE.add_argument('--number', type=int, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)