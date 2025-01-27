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
    fzb_folder = os.path.join(folder, 'FZB/')
    figure_folder = os.path.join(folder, 'FIGURE/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Lens
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        z_phot_lens = file['z_phot'][:].astype(numpy.float32)
        z_spec_lens = file['z_spec'][:].astype(numpy.float32)
    
    # Source
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        z_phot_source = file['z_phot'][:].astype(numpy.float32)
        z_spec_source = file['z_spec'][:].astype(numpy.float32)
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    sigma1 = 1e-4
    sigma2 = 1e+0
    sigma_grid = numpy.geomspace(sigma1, sigma2, grid_size + 1)
    
    delta_lens = numpy.abs(z_phot_lens - z_spec_lens)
    delta_source = numpy.abs(z_phot_source - z_spec_source)
    
    sigma_lens = delta_lens / (1 + z_spec_lens)
    sigma_source = delta_source / (1 + z_spec_source)
    
    print('Lens: {} {}'.format(len(sigma_lens[sigma_lens > 0.15]) / len(sigma_lens) * 100, len(delta_lens[delta_lens > 1.0]) / len(delta_lens) * 100))
    
    print('Source: {} {}'.format(len(sigma_source[sigma_source > 0.15]) / len(sigma_source) * 100, len(delta_source[delta_source > 1.0]) / len(delta_source) * 100))
    
    # Plot
    figure = pyplot.figure(figsize=(15, 12))
    normalize = colors.LogNorm(vmin=1, vmax=20000)
    plot = gridspec.GridSpec(nrows=2, ncols=2, figure=figure, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Plot 1
    plot1 = figure.add_subplot(plot[0, 0])
    
    z_mesh = plot1.hist2d(x=z_phot_lens, y=z_spec_lens, bins=[z_grid, z_grid], norm=normalize, cmap='plasma')[-1]
    
    plot1.plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot1.set_xlim(z1, z2)
    plot1.set_ylim(z1, z2)
    
    plot1.set_xticklabels([])
    plot1.set_ylabel(r'$z_\mathrm{spec}$')
    plot1.get_yticklabels()[0].set_visible(False)
    
    # Plot 2
    plot2 = figure.add_subplot(plot[0, 1])
    
    z_mesh = plot2.hist2d(x=z_phot_source, y=z_spec_source, bins=[z_grid, z_grid], norm=normalize, cmap='plasma')[-1]
    
    plot2.plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot2.set_xlim(z1, z2)
    plot2.set_ylim(z1, z2)
    plot2.set_xlabel(r'$z_\mathrm{phot}$')
    
    plot2.set_xticklabels([])
    plot2.set_yticklabels([])
    
    # Plot 3
    plot3 = figure.add_subplot(plot[1, 0])
    
    z_mesh = plot3.hist2d(x=z_phot_lens, y=sigma_lens, bins=[z_grid, sigma_grid], norm=normalize, cmap='plasma')[-1]
    
    plot3.plot(z_grid, 0.03 * numpy.ones(grid_size + 1), color='black', linestyle='--', linewidth=2.0)
    
    plot3.set_ylim(sigma1, sigma2)
    plot3.set_xlim(z1, z2)
    
    plot3.set_yscale('log')
    plot3.set_xlabel(r'$z_\mathrm{phot}$')
    plot3.set_ylabel(r'$\left| z_\mathrm{phot} - z_\mathrm{spec} \right| / \left(1 + z_\mathrm{spec} \right)$')
    
    # Plot 4
    plot4 = figure.add_subplot(plot[1, 1])
    
    z_mesh = plot4.hist2d(x=z_phot_source, y=sigma_source, bins=[z_grid, sigma_grid], norm=normalize, cmap='plasma')[-1]
    
    plot4.plot(z_grid, 0.05 * numpy.ones(grid_size + 1), color='black', linestyle='--', linewidth=2.0)
    
    plot4.set_yscale('log')
    plot4.set_ylim(sigma1, sigma2)
    plot4.set_xlim(z1, z2)
    
    plot4.set_yticklabels([])
    plot4.set_xlabel(r'$z_\mathrm{phot}$')
    plot4.get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
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
    PARSE = argparse.ArgumentParser(description='Figure Redshift')
    PARSE.add_argument('--tag', type=str, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)