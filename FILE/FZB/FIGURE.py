import os
import time
import numpy
import argparse
from rail import core
from matplotlib import pyplot, colors, gridspec


def plot_select(z_lens, z_source, z_mean, mag_source):
    '''
    Plot the selection of lens and source samples.
    
    Parameters:
        z_lens (list): The redshift range of lens samples.
        z_source (list): The redshift range of source samples.
        z_mean (numpy.ndarray): The mean redshifts of source samples.
        mag_source (numpy.ndarray): The i band magnitudes of source samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    '''
    # Set variables
    mag1 = 16.0
    mag2 = 26.0
    grid_size = 100
    mag_grid = numpy.linspace(mag1, mag2, grid_size + 1)
    
    slope = 4.0
    intersection = 18.0
    
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    
    z_lens_grid = numpy.linspace(z1_lens, z2_lens, grid_size + 1)
    z_source_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
    z_mesh =plot.hist2d(x=z_mean, y=mag_source, bins=[z_source_grid, mag_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot.plot(z_lens_grid, slope * z_lens_grid + intersection, color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(grid_size) * z1_lens, numpy.linspace(mag1, slope * z1_lens + intersection, grid_size), color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(grid_size) * z2_lens, numpy.linspace(mag1, slope * z2_lens + intersection, grid_size), color='black', linestyle='--', linewidth=2.0)
    
    plot.set_ylim(mag1, mag2)
    plot.set_xlim(z1_source, z2_source)
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$z_\mathrm{phot}$')
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00, hspace=0.00)
    return figure


def plot_redshift(z_lens, z_source, z_mean, z_true, mag_source):
    '''
    Plot the photometric redshift distribution of lens and source samples.
    
    Parameters:
        z_lens (list): The redshift range of lens samples.
        z_source (list): The redshift range of source samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_true (numpy.ndarray): The true redshift of source samples.
        mag_source (numpy.ndarray): The i band magnitudes of source samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    '''
    # Set variables
    grid_size = 100
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    slope = 4.0
    intersection = 18.0
    
    sigma1 = 1e-4
    sigma2 = 1e+0
    sigma_grid = numpy.geomspace(sigma1, sigma2, grid_size + 1)
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intersection)
    
    delta_lens = numpy.abs(z_mean[select_lens] - z_true[select_lens])
    delta_source = numpy.abs(z_mean[select_source] - z_true[select_source])
    
    sigma_lens = delta_lens / (1 + z_true[select_lens])
    sigma_source = delta_source / (1 + z_true[select_source])
    
    print('Lens: {} {}'.format(len(sigma_lens[sigma_lens > 0.15]) / len(sigma_lens) * 100, len(delta_lens[delta_lens > 1.0]) / len(delta_lens) * 100))
    
    print('Source: {} {}'.format(len(sigma_source[sigma_source > 0.15]) / len(sigma_source) * 100, len(delta_source[delta_source > 1.0]) / len(delta_source) * 100))
    
    # Plot
    figure = pyplot.figure(figsize=(15, 12))
    
    plot = gridspec.GridSpec(nrows=2, ncols=2, figure=figure, height_ratios=[3, 1], width_ratios=[1, 1])
    
    # Plot 1
    plot1 = figure.add_subplot(plot[0, 0])
    
    z_mesh = plot1.hist2d(x=z_mean[select_lens], y=z_true[select_lens], bins=[z_grid, z_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot1.plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot1.set_xlim(z1_source, z2_source)
    plot1.set_ylim(z1_source, z2_source)
    
    plot1.set_ylabel(r'$z_\mathrm{spec}$')
    plot1.get_xticklabels()[0].set_visible(False)
    plot1.get_yticklabels()[0].set_visible(False)
    
    # Plot 2
    plot2 = figure.add_subplot(plot[0, 1])
    
    z_mesh = plot2.hist2d(x=z_mean[select_source], y=z_true[select_source], bins=[z_grid, z_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot2.plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot2.set_xlim(z1_source, z2_source)
    plot2.set_ylim(z1_source, z2_source)
    plot2.set_xlabel(r'$z_\mathrm{phot}$')
    
    plot2.set_yticklabels([])
    plot2.get_xticklabels()[0].set_visible(False)
    
    # Plot 3
    plot3 = figure.add_subplot(plot[1, 0])
    
    z_mesh = plot3.hist2d(x=z_mean[select_lens], y=sigma_lens, bins=[z_grid, sigma_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot3.plot(z_grid, 0.03 * numpy.ones(grid_size + 1), color='black', linestyle='--', linewidth=2.0)
    
    plot3.set_ylim(sigma1, sigma2)
    plot3.set_xlim(z1_source, z2_source)
    
    plot3.set_yscale('log')
    plot3.set_xlabel(r'$z_\mathrm{phot}$')
    plot3.set_ylabel(r'$\left| z_\mathrm{phot} - z_\mathrm{spec} \right| / \left(1 + z_\mathrm{spec} \right)$')
    
    # Plot 4
    plot4 = figure.add_subplot(plot[1, 1])
    
    z_mesh = plot4.hist2d(x=z_mean[select_source], y=sigma_source, bins=[z_grid, sigma_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot4.plot(z_grid, 0.05 * numpy.ones(grid_size + 1), color='black', linestyle='--', linewidth=2.0)
    
    plot4.set_yscale('log')
    plot4.set_ylim(sigma1, sigma2)
    plot4.set_xlim(z1_source, z2_source)
    
    plot4.set_xticklabels([])
    plot4.set_yticklabels([])
    
    plot4.set_xlabel(r'$z_\mathrm{phot}$')
    plot4.get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00, hspace=0.00)
    return figure


def main(index, folder):
    '''
    Plot the figures of the FZB modelling.
    
    Arguments:
        index (int): The number of datasets.
        folder (str): The base folder of the datasets.
    
    Returns:
        float: The duration of the plotter.
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(fzb_folder, 'FIGURE/'), exist_ok=True)
    
    # Load 
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Load
    application_name = os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    z_true = application_dataset['photometry']['redshift']
    mag_source = application_dataset['photometry']['mag_i_lsst']
    color_source = application_dataset['photometry']['mag_g_lsst'] - application_dataset['photometry']['mag_z_lsst']
    
    del application_dataset
    
    estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_mean = numpy.concatenate(estimator.mean())
    del estimator
    indices = numpy.isclose(z_mean, 1.50, rtol=0.001)
    print(z_true[indices], z_true[indices].min(), z_true[indices].max())
    print(mag_source[indices], mag_source[indices].min(), mag_source[indices].max())
    print(color_source[indices], color_source[indices].min(), color_source[indices].max())
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    z_lens = [z1_lens, z2_lens]
    
    z1_source = 0.0
    z2_source = 3.0
    z_source = [z1_source, z2_source]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Plot
    os.makedirs(os.path.join(fzb_folder, 'FIGURE/SELECT'), exist_ok=True)
    figure = plot_select(z_lens, z_source, z_mean, mag_source)
    figure.savefig(os.path.join(fzb_folder, 'FIGURE/SELECT/FIGURE{}.png'.format(index)), bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    os.makedirs(os.path.join(fzb_folder, 'FIGURE/REDSHIFT'), exist_ok=True)
    figure = plot_redshift(z_lens, z_source, z_mean, z_true, mag_source)
    figure.savefig(os.path.join(fzb_folder, 'FIGURE/REDSHIFT/FIGURE{}.png'.format(index)), bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Tomography Binning')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)