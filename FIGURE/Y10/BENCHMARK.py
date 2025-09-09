import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot, colors, gridspec


def main(tag, index, folder):
    '''
    Plot the figures of the benchmark redshift estimation
    
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
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/BENCHMARK/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    bin_size = 100
    z_bin = numpy.linspace(z1, z2, bin_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_redshift_true = file['photometry']['redshift_true'][...]
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        z_phot = file['z_phot'][...]
        reference_lens = file['reference_lens'][...]
        reference_source = file['reference_source'][...]
    
    # Lens
    z_phot_lens = z_phot[reference_lens]
    z_true_lens = combination_redshift_true[reference_lens]
    
    # Source
    z_phot_source = z_phot[reference_source]
    z_true_source = combination_redshift_true[reference_source]
    
    # Delta
    delta1 = 1e-4
    delta2 = 1e+0
    delta_bin = numpy.geomspace(delta1, delta2, bin_size + 1)
    
    delta_lens = numpy.abs(z_phot_lens - z_true_lens) / (1 + z_true_lens)
    delta_source = numpy.abs(z_phot_source - z_true_source) / (1 + z_true_source)
    
    # Mean
    width = 0.2
    mean_size = int((z2 - z1) / width) + 1
    z_mean = numpy.linspace(z1 - width / 2, z2 + width / 2, mean_size + 1)
    
    delta_mean_lens = numpy.zeros(mean_size)
    delta_mean_source = numpy.zeros(mean_size)
    
    for m in range(mean_size):
        reference_lens = (z_mean[m] <= z_true_lens) & (z_true_lens < z_mean[m + 1])
        if numpy.sum(reference_lens) > 0:
            delta_mean_lens[m] = numpy.median(delta_lens[reference_lens])
        
        reference_source = (z_mean[m] <= z_true_source) & (z_true_source < z_mean[m + 1])
        if numpy.sum(reference_source) > 0:
            delta_mean_source[m] = numpy.median(delta_source[reference_source])
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 30
    
    figure = pyplot.figure(figsize=(15, 10))
    normalize = colors.LogNorm(vmin=1, vmax=1000)
    plot = gridspec.GridSpec(nrows=2, ncols=2, figure=figure, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Plot 1
    plot1 = figure.add_subplot(plot[0, 0])
    
    image = plot1.hist2d(x=z_true_lens, y=z_phot_lens, bins=[z_bin, z_bin], norm=normalize, cmap='turbo')[-1]
    
    plot1.plot(z_bin, z_bin - 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot1.plot(z_bin, z_bin + 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot1.plot(z_bin, z_bin, color='black', linestyle='-', linewidth=2.5)
    
    plot1.set_xlim(z1, z2)
    plot1.set_ylim(z1, z2)
    
    plot1.set_xticklabels([])
    plot1.get_yticklabels()[0].set_visible(False)
    
    plot1.set_title(r'$\mathrm{Lens}$')
    plot1.set_ylabel(r'$z_\mathrm{phot}$')
    
    # Plot 2
    plot2 = figure.add_subplot(plot[0, 1])
    
    image = plot2.hist2d(x=z_true_source, y=z_phot_source, bins=[z_bin, z_bin], norm=normalize, cmap='turbo')[-1]
    
    plot2.plot(z_bin, z_bin - 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot2.plot(z_bin, z_bin + 0.15 * (1 + z_bin), color='black', linestyle='-.', linewidth=2.5)
    
    plot2.plot(z_bin, z_bin, color='black', linestyle='-', linewidth=2.5)
    
    plot2.set_xlim(z1, z2)
    plot2.set_ylim(z1, z2)
    
    plot2.set_xticklabels([])
    plot2.set_yticklabels([])
    plot2.set_title(r'$\mathrm{Source}$')
    
    # Plot 3
    plot3 = figure.add_subplot(plot[1, 0])
    
    image = plot3.hist2d(x=z_true_lens, y=delta_lens, bins=[z_bin, delta_bin], norm=normalize, cmap='turbo')[-1]
    
    plot3.plot(z_bin, 0.03 * numpy.ones(bin_size + 1), color='black', linestyle=':', linewidth=2.5)
    
    plot3.plot((z_mean[1:] + z_mean[:-1]) / 2, delta_mean_lens, color='black', linestyle='--', linewidth=2.5)
    
    plot3.set_xlim(z1, z2)
    plot3.set_ylim(delta1, delta2)
    
    plot3.set_yscale('log')
    plot3.set_xlabel(r'$z_\mathrm{true}$')
    plot3.set_ylabel(r'$\left| \delta_z \right|$')
    
    # Plot 4
    plot4 = figure.add_subplot(plot[1, 1])
    
    image = plot4.hist2d(x=z_true_source, y=delta_source, bins=[z_bin, delta_bin], norm=normalize, cmap='turbo')[-1]
    
    plot4.plot(z_bin, 0.05 * numpy.ones(bin_size + 1), color='black', linestyle=':', linewidth=2.5)
    
    plot4.plot((z_mean[1:] + z_mean[:-1]) / 2, delta_mean_source, color='black', linestyle='--', linewidth=2.5)
    
    plot4.set_xlim(z1, z2)
    plot4.set_ylim(delta1, delta2)
    plot4.set_xlabel(r'$z_\mathrm{true}$')
    
    plot4.set_yscale('log')
    plot4.set_yticklabels([])
    plot4.get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.92, 0.15, 0.03, 0.70]), orientation='vertical')
    figure.subplots_adjust(right=0.90, wspace=0.00, hspace=0.00)
    color_bar.set_label(r'$\mathrm{Counts}$', fontsize=25)
    
    # Save    
    figure.savefig(os.path.join(figure_folder, '{}/BENCHMARK/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure Benchmark')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)