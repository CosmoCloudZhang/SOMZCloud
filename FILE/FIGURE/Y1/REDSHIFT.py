import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot, colors, gridspec


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
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    
    bin_size = 100
    z_bin = numpy.linspace(z1, z2, bin_size + 1)
    
    # Sigma
    sigma1 = 1e-4
    sigma2 = 1e+0
    sigma_bin = numpy.geomspace(sigma1, sigma2, bin_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_redshift_true = file['photometry']['redshift_true'][...]
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        z_phot = file['z_phot'][...]
        select_lens = file['select_lens'][...]
        select_source = file['select_source'][...]
    
    # Lens
    z_phot_lens = z_phot[select_lens]
    z_true_lens = application_redshift_true[select_lens]
    
    # Source
    z_phot_source = z_phot[select_source]
    z_true_source = application_redshift_true[select_source]
    
    # Sigma
    sigma_lens = numpy.abs(z_phot_lens - z_true_lens) / (1 + z_true_lens)
    sigma_source = numpy.abs(z_phot_source - z_true_source) / (1 + z_true_source)
    
    sigma_mean_lens = numpy.zeros(bin_size)
    sigma_mean_source = numpy.zeros(bin_size)
    
    for m in range(bin_size):
        select_lens = (z_bin[m] <= z_phot_lens) & (z_phot_lens < z_bin[m + 1])
        if numpy.sum(select_lens) > 0:
            sigma_mean_lens[m] = numpy.mean(sigma_lens[select_lens])
        
        select_source = (z_bin[m] <= z_phot_source) & (z_phot_source < z_bin[m + 1])
        if numpy.sum(select_source) > 0:
            sigma_mean_source[m] = numpy.mean(sigma_source[select_source])
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure = pyplot.figure(figsize=(15, 12))
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    plot = gridspec.GridSpec(nrows=2, ncols=2, figure=figure, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Plot 1
    plot1 = figure.add_subplot(plot[0, 0])
    
    image = plot1.hist2d(x=z_phot_lens, y=z_true_lens, bins=[z_bin, z_bin], norm=normalize, cmap='plasma')[-1]
    
    plot1.plot(z_bin, z_bin - 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot1.plot(z_bin, z_bin + 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot1.plot(z_bin, z_bin, color='black', linestyle='-', linewidth=2.5)
    
    plot1.set_xlim(z1, z2)
    plot1.set_ylim(z1, z2)
    
    plot1.set_xticklabels([])
    plot1.set_ylabel(r'$z_\mathrm{true}$')
    plot1.get_yticklabels()[0].set_visible(False)
    
    # Plot 2
    plot2 = figure.add_subplot(plot[0, 1])
    
    image = plot2.hist2d(x=z_phot_source, y=z_true_source, bins=[z_bin, z_bin], norm=normalize, cmap='plasma')[-1]
    
    plot2.plot(z_bin, z_bin - 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot2.plot(z_bin, z_bin + 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot2.plot(z_bin, z_bin, color='black', linestyle='-', linewidth=2.5)
    
    plot2.set_xlim(z1, z2)
    plot2.set_ylim(z1, z2)
    plot2.set_xlabel(r'$z_\mathrm{phot}$')
    
    plot2.set_xticklabels([])
    plot2.set_yticklabels([])
    
    # Plot 3
    plot3 = figure.add_subplot(plot[1, 0])
    
    image = plot3.hist2d(x=z_phot_lens, y=sigma_lens, bins=[z_bin, sigma_bin], norm=normalize, cmap='plasma')[-1]
    
    plot3.plot((z_bin[1:] + z_bin[:-1]) / 2, sigma_mean_lens, color='black', linestyle='--', linewidth=2.5)
    
    plot3.plot(z_bin, 0.03 * numpy.ones(bin_size + 1), color='black', linestyle=':', linewidth=2.5)
    
    plot3.set_xlim(z1, z2)
    plot3.set_ylim(sigma1, sigma2)
    
    plot3.set_yscale('log')
    plot3.set_ylabel(r'$\sigma_z$')
    plot3.set_xlabel(r'$z_\mathrm{phot}$')
    
    # Plot 4
    plot4 = figure.add_subplot(plot[1, 1])
    
    image = plot4.hist2d(x=z_phot_source, y=sigma_source, bins=[z_bin, sigma_bin], norm=normalize, cmap='plasma')[-1]
    
    plot4.plot((z_bin[1:] + z_bin[:-1]) / 2, sigma_mean_source, color='black', linestyle='--', linewidth=2.5)
    
    plot4.plot(z_bin, 0.05 * numpy.ones(bin_size + 1), color='black', linestyle=':', linewidth=2.5)
    
    plot4.set_xlim(z1, z2)
    plot4.set_ylim(sigma1, sigma2)
    plot4.set_xlabel(r'$z_\mathrm{phot}$')
    
    plot4.set_yscale('log')
    plot4.set_yticklabels([])
    plot4.get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.20, wspace=0.00, hspace=0.00)
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/REDSHIFT/'.format(tag)), exist_ok=True)
    
    figure.savefig(os.path.join(figure_folder, '{}/REDSHIFT/FIGURE{}.png'.format(tag, index)), format='png', bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Sample')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)