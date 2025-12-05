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
    compare_folder = os.path.join(folder, 'COMPARE/')
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/METRIC/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1_average_lens = 0.0
    z2_average_lens = 1.5
    average_size_lens = 5
    average_bias_lens = (z2_average_lens - z1_average_lens) / average_size_lens
    z_average_lens = numpy.linspace(z1_average_lens + average_bias_lens / 2, z2_average_lens - average_bias_lens / 2, average_size_lens)
    
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_size_source = 6
    average_bias_source = (z2_average_source - z1_average_source) / average_size_source
    z_average_source = numpy.linspace(z1_average_source + average_bias_source / 2, z2_average_source - average_bias_source / 2, average_size_source)
    
    # Average
    bias_lens = numpy.zeros((number + 1, average_size_lens))
    bias_source = numpy.zeros((number + 1, average_size_source))
    
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
        with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            
            bias_lens[index, :] = file['bias_lens'][...]
            bias_source[index, :] = file['bias_source'][...]
            
            sigma_lens[index, :] = file['sigma_lens'][...]
            sigma_source[index, :] = file['sigma_source'][...]
            
            fraction_lens[index, :] = file['fraction_lens'][...]
            fraction_source[index, :] = file['fraction_source'][...]
            
            rate_lens[index, :] = file['rate_lens'][...]
            rate_source[index, :] = file['rate_source'][...]
            
            divergence_lens[index, :] = file['divergence_lens'][...]
            divergence_source[index, :] = file['divergence_source'][...]
    
    # Reference
    reference_bias_lens = numpy.zeros((number + 1, average_size_lens))
    reference_bias_source = numpy.zeros((number + 1, average_size_source))
    
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
            reference_bias_lens[index, :] = file['bias_lens'][...]
            reference_bias_source[index, :] = file['bias_source'][...]
            
            reference_sigma_lens[index, :] = file['sigma_lens'][...]
            reference_sigma_source[index, :] = file['sigma_source'][...]
            
            reference_fraction_lens[index, :] = file['fraction_lens'][...]
            reference_fraction_source[index, :] = file['fraction_source'][...]
            
            reference_rate_lens[index, :] = file['rate_lens'][...]
            reference_rate_source[index, :] = file['rate_source'][...]
            
            reference_divergence_lens[index, :] = file['divergence_lens'][...]
            reference_divergence_source[index, :] = file['divergence_source'][...]
    
    # Compare
    compare_bias_lens = numpy.zeros((number + 1, average_size_lens))
    compare_bias_source = numpy.zeros((number + 1, average_size_source))
    
    compare_sigma_lens = numpy.zeros((number + 1, average_size_lens))
    compare_sigma_source = numpy.zeros((number + 1, average_size_source))
    
    compare_fraction_lens = numpy.zeros((number + 1, average_size_lens))
    compare_fraction_source = numpy.zeros((number + 1, average_size_source))
    
    compare_rate_lens = numpy.zeros((number + 1, average_size_lens))
    compare_rate_source = numpy.zeros((number + 1, average_size_source))
    
    compare_divergence_lens = numpy.zeros((number + 1, average_size_lens))
    compare_divergence_source = numpy.zeros((number + 1, average_size_source))
    
    # Loop
    for index in range(number + 1):
        with h5py.File(os.path.join(compare_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            compare_bias_lens[index, :] = file['bias_lens'][...]
            compare_bias_source[index, :] = file['bias_source'][...]
            
            compare_sigma_lens[index, :] = file['sigma_lens'][...]
            compare_sigma_source[index, :] = file['sigma_source'][...]
            
            compare_fraction_lens[index, :] = file['fraction_lens'][...]
            compare_fraction_source[index, :] = file['fraction_source'][...]
            
            compare_rate_lens[index, :] = file['rate_lens'][...]
            compare_rate_source[index, :] = file['rate_source'][...]
            
            compare_divergence_lens[index, :] = file['divergence_lens'][...]
            compare_divergence_source[index, :] = file['divergence_source'][...]
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure = pyplot.figure(figsize=(18, 24))
    plot_list = gridspec.GridSpec(nrows=5, ncols=2, figure=figure)
    
    # Plot lens delta 
    plot = figure.add_subplot(plot_list[0, 0])
    
    plot.fill_between(x=numpy.linspace(z1_average_lens, z2_average_lens, average_size_lens + 1), y1=numpy.ones(average_size_lens + 1) * -0.02, y2=numpy.ones(average_size_lens + 1) * +0.02, color='grey', alpha=0.5)
    
    plot.errorbar(x=z_average_lens, y=numpy.median(bias_lens, axis=0), yerr=[numpy.median(bias_lens, axis=0) - numpy.quantile(bias_lens, 0.16, axis=0), numpy.quantile(bias_lens, 0.84, axis=0) - numpy.median(bias_lens, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(compare_bias_lens, axis=0), yerr=[numpy.median(compare_bias_lens, axis=0) - numpy.quantile(compare_bias_lens, 0.16, axis=0), numpy.quantile(compare_bias_lens, 0.84, axis=0) - numpy.median(compare_bias_lens, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(reference_bias_lens, axis=0), yerr=[numpy.median(reference_bias_lens, axis=0) - numpy.quantile(reference_bias_lens, 0.16, axis=0), numpy.quantile(reference_bias_lens, 0.84, axis=0) - numpy.median(reference_bias_lens, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.060, +0.060)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{Lens}$', fontsize=35)
    plot.set_ylabel(r'$\bar{\delta}_z$', fontsize=35)
    
    # Plot lens sigma 
    plot = figure.add_subplot(plot_list[1, 0])
    
    plot.fill_between(x=numpy.linspace(z1_average_lens, z2_average_lens, average_size_lens + 1), y1=numpy.zeros(average_size_lens + 1), y2=numpy.ones(average_size_lens + 1) * 0.02, color='grey', alpha=0.5)
    
    plot.errorbar(x=z_average_lens, y=numpy.median(sigma_lens, axis=0), yerr=[numpy.median(sigma_lens, axis=0) - numpy.quantile(sigma_lens, 0.16, axis=0), numpy.quantile(sigma_lens, 0.84, axis=0) - numpy.median(sigma_lens, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(compare_sigma_lens, axis=0), yerr=[numpy.median(compare_sigma_lens, axis=0) - numpy.quantile(compare_sigma_lens, 0.16, axis=0), numpy.quantile(compare_sigma_lens, 0.84, axis=0) - numpy.median(compare_sigma_lens, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(reference_sigma_lens, axis=0), yerr=[numpy.median(reference_sigma_lens, axis=0) - numpy.quantile(reference_sigma_lens, 0.16, axis=0), numpy.quantile(reference_sigma_lens, 0.84, axis=0) - numpy.median(reference_sigma_lens, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.006, +0.060)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$', fontsize=35)
    
    # Plot lens fraction
    plot = figure.add_subplot(plot_list[2, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.median(fraction_lens, axis=0), yerr=[numpy.median(fraction_lens, axis=0) - numpy.quantile(fraction_lens, 0.16, axis=0), numpy.quantile(fraction_lens, 0.84, axis=0) - numpy.median(fraction_lens, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(compare_fraction_lens, axis=0), yerr=[numpy.median(compare_fraction_lens, axis=0) - numpy.quantile(compare_fraction_lens, 0.16, axis=0), numpy.quantile(compare_fraction_lens, 0.84, axis=0) - numpy.median(compare_fraction_lens, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(reference_fraction_lens, axis=0), yerr=[numpy.median(reference_fraction_lens, axis=0) - numpy.quantile(reference_fraction_lens, 0.16, axis=0), numpy.quantile(reference_fraction_lens, 0.84, axis=0) - numpy.median(reference_fraction_lens, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.003, +0.030)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$', fontsize=35)
    
    # Plot lens rate
    plot = figure.add_subplot(plot_list[3, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.median(rate_lens, axis=0), yerr=[numpy.median(rate_lens, axis=0) - numpy.quantile(rate_lens, 0.16, axis=0), numpy.quantile(rate_lens, 0.84, axis=0) - numpy.median(rate_lens, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(compare_rate_lens, axis=0), yerr=[numpy.median(compare_rate_lens, axis=0) - numpy.quantile(compare_rate_lens, 0.16, axis=0), numpy.quantile(compare_rate_lens, 0.84, axis=0) - numpy.median(compare_rate_lens, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(reference_rate_lens, axis=0), yerr=[numpy.median(reference_rate_lens, axis=0) - numpy.quantile(reference_rate_lens, 0.16, axis=0), numpy.quantile(reference_rate_lens, 0.84, axis=0) - numpy.median(reference_rate_lens, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.002, +0.020)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$', fontsize=35)
    
    # Plot lens divergence
    plot = figure.add_subplot(plot_list[4, 0])
    
    plot.errorbar(x=z_average_lens, y=numpy.median(divergence_lens, axis=0), yerr=[numpy.median(divergence_lens, axis=0) - numpy.quantile(divergence_lens, 0.16, axis=0), numpy.quantile(divergence_lens, 0.84, axis=0) - numpy.median(divergence_lens, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(compare_divergence_lens, axis=0), yerr=[numpy.median(compare_divergence_lens, axis=0) - numpy.quantile(compare_divergence_lens, 0.16, axis=0), numpy.quantile(compare_divergence_lens, 0.84, axis=0) - numpy.median(compare_divergence_lens, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_lens, y=numpy.median(reference_divergence_lens, axis=0), yerr=[numpy.median(reference_divergence_lens, axis=0) - numpy.quantile(reference_divergence_lens, 0.16, axis=0), numpy.quantile(reference_divergence_lens, 0.84, axis=0) - numpy.median(reference_divergence_lens, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.400, +4.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_ylabel(r'$\mathcal{D}_q$', fontsize=35)
    plot.set_xlabel(r'$z_\mathrm{true}$', fontsize=35)
    
    # Plot source delta 
    plot = figure.add_subplot(plot_list[0, 1])
    
    plot.fill_between(x=numpy.linspace(z1_average_source, z2_average_source, average_size_source + 1), y1=numpy.ones(average_size_source + 1) * -0.03, y2=numpy.ones(average_size_source + 1) * +0.03, color='grey', alpha=0.5)
    
    plot.errorbar(x=z_average_source, y=numpy.median(bias_source, axis=0), yerr=[numpy.median(bias_source, axis=0) - numpy.quantile(bias_source, 0.16, axis=0), numpy.quantile(bias_source, 0.84, axis=0) - numpy.median(bias_source, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(compare_bias_source, axis=0), yerr=[numpy.median(compare_bias_source, axis=0) - numpy.quantile(compare_bias_source, 0.16, axis=0), numpy.quantile(compare_bias_source, 0.84, axis=0) - numpy.median(compare_bias_source, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(reference_bias_source, axis=0), yerr=[numpy.median(reference_bias_source, axis=0) - numpy.quantile(reference_bias_source, 0.16, axis=0), numpy.quantile(reference_bias_source, 0.84, axis=0) - numpy.median(reference_bias_source, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-1.000, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{Source}$', fontsize=35)
    plot.set_ylabel(r'$\bar{\delta}_z$', fontsize=35)
    
    # Plot source sigma 
    plot = figure.add_subplot(plot_list[1, 1])
    
    plot.fill_between(x=numpy.linspace(z1_average_source, z2_average_source, average_size_source + 1), y1=numpy.zeros(average_size_source + 1), y2=numpy.ones(average_size_source + 1) * 0.03, color='grey', alpha=0.5)
    
    plot.errorbar(x=z_average_source, y=numpy.median(sigma_source, axis=0), yerr=[numpy.median(sigma_source, axis=0) - numpy.quantile(sigma_source, 0.16, axis=0), numpy.quantile(sigma_source, 0.84, axis=0) - numpy.median(sigma_source, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(compare_sigma_source, axis=0), yerr=[numpy.median(compare_sigma_source, axis=0) - numpy.quantile(compare_sigma_source, 0.16, axis=0), numpy.quantile(compare_sigma_source, 0.84, axis=0) - numpy.median(compare_sigma_source, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(reference_sigma_source, axis=0), yerr=[numpy.median(reference_sigma_source, axis=0) - numpy.quantile(reference_sigma_source, 0.16, axis=0), numpy.quantile(reference_sigma_source, 0.84, axis=0) - numpy.median(reference_sigma_source, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.036, +0.360)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$', fontsize=35)
    
    # Plot source fraction
    plot = figure.add_subplot(plot_list[2, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.median(fraction_source, axis=0), yerr=[numpy.median(fraction_source, axis=0) - numpy.quantile(fraction_source, 0.16, axis=0), numpy.quantile(fraction_source, 0.84, axis=0) - numpy.median(fraction_source, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(compare_fraction_source, axis=0), yerr=[numpy.median(compare_fraction_source, axis=0) - numpy.quantile(compare_fraction_source, 0.16, axis=0), numpy.quantile(compare_fraction_source, 0.84, axis=0) - numpy.median(compare_fraction_source, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(reference_fraction_source, axis=0), yerr=[numpy.median(reference_fraction_source, axis=0) - numpy.quantile(reference_fraction_source, 0.16, axis=0), numpy.quantile(reference_fraction_source, 0.84, axis=0) - numpy.median(reference_fraction_source, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.120, +1.200)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$', fontsize=35)
    
    # Plot source rate
    plot = figure.add_subplot(plot_list[3, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.median(rate_source, axis=0), yerr=[numpy.median(rate_source, axis=0) - numpy.quantile(rate_source, 0.16, axis=0), numpy.quantile(rate_source, 0.84, axis=0) - numpy.median(rate_source, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(compare_rate_source, axis=0), yerr=[numpy.median(compare_rate_source, axis=0) - numpy.quantile(compare_rate_source, 0.16, axis=0), numpy.quantile(compare_rate_source, 0.84, axis=0) - numpy.median(compare_rate_source, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(reference_rate_source, axis=0), yerr=[numpy.median(reference_rate_source, axis=0) - numpy.quantile(reference_rate_source, 0.16, axis=0), numpy.quantile(reference_rate_source, 0.84, axis=0) - numpy.median(reference_rate_source, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.120, +1.200)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$', fontsize=35)
    
    # Plot source divergence
    plot = figure.add_subplot(plot_list[4, 1])
    
    plot.errorbar(x=z_average_source, y=numpy.median(divergence_source, axis=0), yerr=[numpy.median(divergence_source, axis=0) - numpy.quantile(divergence_source, 0.16, axis=0), numpy.quantile(divergence_source, 0.84, axis=0) - numpy.median(divergence_source, axis=0)], color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(compare_divergence_source, axis=0), yerr=[numpy.median(compare_divergence_source, axis=0) - numpy.quantile(compare_divergence_source, 0.16, axis=0), numpy.quantile(compare_divergence_source, 0.84, axis=0) - numpy.median(compare_divergence_source, axis=0)], color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Application} \, \mathrm{trained \, on} \, \mathtt{Degradation}$')
    
    plot.errorbar(x=z_average_source, y=numpy.median(reference_divergence_source, axis=0), yerr=[numpy.median(reference_divergence_source, axis=0) - numpy.quantile(reference_divergence_source, 0.16, axis=0), numpy.quantile(reference_divergence_source, 0.84, axis=0) - numpy.median(reference_divergence_source, axis=0)], color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, capsize=5, capthick=2.5, alpha=0.8, label=r'$\mathtt{Combination} \, \mathrm{trained \, on} \, \mathtt{Combination}$')
    
    plot.set_ylim(-0.500, +5.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_ylabel(r'$\mathcal{D}_q$', fontsize=35)
    plot.set_xlabel(r'$z_\mathrm{true}$', fontsize=35)
    
    # Legend
    handles, labels = plot.get_legend_handles_labels()
    figure.legend(handles, labels, loc='center', ncol=1, fontsize=35, bbox_to_anchor=(0.5, 0.0), frameon=True)
    
    # Save
    figure.subplots_adjust(wspace=0.24, hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/METRIC/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
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