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
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        
        delta_lens = file['delta_lens'][...]
        delta_source = file['delta_source'][...]
        
        sigma_lens = file['sigma_lens'][...]
        sigma_source = file['sigma_source'][...]
        
        fraction_lens = file['fraction_lens'][...]
        fraction_source = file['fraction_source'][...]
        
        rate_lens = file['rate_lens'][...]
        rate_source = file['rate_source'][...]
        
        divergence_lens = file['divergence_lens'][...]
        divergence_source = file['divergence_source'][...]
        
        score_lens = file['score_lens'][...]
        score_source = file['score_source'][...]
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        reference_sigma_lens = file['sigma_lens'][...]
        reference_sigma_source = file['sigma_source'][...]
        
        reference_delta_lens = file['delta_lens'][...]
        reference_delta_source = file['delta_source'][...]
        
        reference_fraction_lens = file['fraction_lens'][...]
        reference_fraction_source = file['fraction_source'][...]
        
        reference_rate_lens = file['rate_lens'][...]
        reference_rate_source = file['rate_source'][...]
        
        reference_divergence_lens = file['divergence_lens'][...]
        reference_divergence_source = file['divergence_source'][...]
        
        reference_score_lens = file['score_lens'][...]
        reference_score_source = file['score_source'][...]
    
    # Comparison
    with h5py.File(os.path.join(comparison_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        comparison_sigma_lens = file['sigma_lens'][...]
        comparison_sigma_source = file['sigma_source'][...]
        
        comparison_delta_lens = file['delta_lens'][...]
        comparison_delta_source = file['delta_source'][...]
        
        comparison_fraction_lens = file['fraction_lens'][...]
        comparison_fraction_source = file['fraction_source'][...]   
        
        comparison_rate_lens = file['rate_lens'][...]
        comparison_rate_source = file['rate_source'][...]
        
        comparison_divergence_lens = file['divergence_lens'][...]
        comparison_divergence_source = file['divergence_source'][...]
        
        comparison_score_lens = file['score_lens'][...]
        comparison_score_source = file['score_source'][...]
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    figure = pyplot.figure(figsize=(16, 24))
    plot_list = gridspec.GridSpec(nrows=6, ncols=2, figure=figure)
    
    # Plot lens delta 
    plot = figure.add_subplot(plot_list[0, 0])
    
    plot.errorbar(x=z_average_lens, y=delta_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_delta_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_delta_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.text(x=0.100, y=0.045, s='$\mathrm{Application \, with \, augmentation}$', color='darkorange', fontsize=25)
    
    plot.text(x=0.100, y=0.035, s='$\mathrm{Combination \, with \, augmentation}$', color='darkgreen', fontsize=25)
    
    plot.text(x=0.100, y=0.025, s='$\mathrm{Application \, without \, augmentation}$', color='darkblue', fontsize=25)
    
    plot.set_ylim(-0.005, +0.050)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{lens}$')
    plot.set_ylabel(r'$\tilde{\delta}_z$')
    
    # Plot lens sigma 
    plot = figure.add_subplot(plot_list[1, 0])
    
    plot.errorbar(x=z_average_lens, y=sigma_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_sigma_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_sigma_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.005, +0.050)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$')
    
    # Plot lens fraction
    plot = figure.add_subplot(plot_list[2, 0])
    
    plot.errorbar(x=z_average_lens, y=fraction_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_fraction_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_fraction_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$')
    
    # Plot lens rate
    plot = figure.add_subplot(plot_list[3, 0])
    
    plot.errorbar(x=z_average_lens, y=rate_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_rate_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_rate_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$')
    
    # Plot lens divergence
    plot = figure.add_subplot(plot_list[4, 0])
    
    plot.errorbar(x=z_average_lens, y=divergence_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_divergence_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_divergence_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.500, +5.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\mathcal{D}_q$')
    
    # Plot lens score
    plot = figure.add_subplot(plot_list[5, 0])
    
    plot.errorbar(x=z_average_lens, y=score_lens, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=reference_score_lens, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_lens, y=comparison_score_lens, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_lens, z2_average_lens)
    
    plot.set_xlabel(r'$z_\mathrm{true}$')
    plot.set_ylabel(r'$\mathcal{L}_q$')
    
    # Plot source delta 
    plot = figure.add_subplot(plot_list[0, 1])
    
    plot.errorbar(x=z_average_source, y=delta_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_delta_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_delta_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_title(r'$\mathrm{source}$')
    plot.set_ylabel(r'$\tilde{\delta}_z$')
    
    # Plot source sigma 
    plot = figure.add_subplot(plot_list[1, 1])
    
    plot.errorbar(x=z_average_source, y=sigma_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_sigma_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_sigma_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.050, +0.500)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\sigma_z$')
    
    # Plot source fraction
    plot = figure.add_subplot(plot_list[2, 1])
    
    plot.errorbar(x=z_average_source, y=fraction_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_fraction_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_fraction_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$f_\mathrm{o}$')
    
    # Plot source rate
    plot = figure.add_subplot(plot_list[3, 1])
    
    plot.errorbar(x=z_average_source, y=rate_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_rate_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_rate_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$r_\mathrm{c}$')
    
    # Plot source divergence
    plot = figure.add_subplot(plot_list[4, 1])
    
    plot.errorbar(x=z_average_source, y=divergence_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_divergence_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_divergence_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.500, +5.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xticklabels([])
    plot.set_ylabel(r'$\mathcal{D}_q$')
    
    # Plot source score
    plot = figure.add_subplot(plot_list[5, 1])
    
    plot.errorbar(x=z_average_source, y=score_source, color='darkorange', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=reference_score_source, color='darkgreen', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.errorbar(x=z_average_source, y=comparison_score_source, color='darkblue', linestyle='-', linewidth=2.5, marker='s', markersize=10, alpha=0.8)
    
    plot.set_ylim(-0.100, +1.000)
    plot.set_xlim(z1_average_source, z2_average_source)
    
    plot.set_xlabel(r'$z_\mathrm{true}$')
    plot.set_ylabel(r'$\mathcal{L}_q$')
    
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