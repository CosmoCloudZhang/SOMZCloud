import os
import time
import numpy
import argparse
from rail import core
import multiprocessing
from matplotlib import pyplot, colors, gridspec

def plot_select(z_grid, z_mean, z_lens, z_source, mag_source):
    
    mag1 = 14.0
    mag2 = 26.0
    mag_size = 100
    mag_grid = numpy.linspace(mag1, mag2, mag_size + 1)
    
    slope = 4.0
    intersection = 18.0
    
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    z_lens_grid = numpy.linspace(z1_lens, z2_lens, z_grid.size + 1)
    
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
    z_mesh =plot.hist2d(x=z_mean, y=mag_source, bins=[z_grid, mag_grid], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot.plot(z_lens_grid, slope * z_lens_grid + intersection, color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(z_grid.size) * z1_lens, numpy.linspace(mag1, slope * z1_lens + intersection, z_grid.size), color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(z_grid.size) * z2_lens, numpy.linspace(mag1, slope * z2_lens + intersection, z_grid.size), color='black', linestyle='--', linewidth=2.0)
    
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


def plot_redshift(z_grid, z_mean, z_true, z_lens, z_source, mag_source):
    """
    Plot the photometric redshift distribution of lens and source samples.
    
    Parameters:
        z_grid (numpy.ndarray): The redshift grid of source samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_true (numpy.ndarray): The redshifts of test application samples.
        z_lens (list): The redshift range of lens samples.
        z_source (list): The redshift range of source samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    
    """
    
    # Set variables
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    
    slope = 4.0
    intersection = 18.0
    
    grid_size = 100
    z_grid = numpy.linspace(z_grid.min(), z_grid.max(), grid_size + 1)
    
    sigma1 = 1e-4
    sigma2 = 1e+0
    sigma_grid = numpy.geomspace(sigma1, sigma2, grid_size + 1)
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intersection)
    
    delta_lens = numpy.abs(z_mean[select_lens] - z_true[select_lens])
    delta_source = numpy.abs(z_mean[select_source] - z_true[select_source])
    
    sigma_lens = delta_lens / (1 + z_true[select_lens])
    sigma_source = delta_source / (1 + z_true[select_source])
    
    print('Lens: {} {}'.format(len(sigma_lens[sigma_lens > 0.1]) / len(sigma_lens) * 100, len(delta_lens[delta_lens > 1.0]) / len(delta_lens) * 100))
    
    print('Source: {} {}'.format(len(sigma_source[sigma_source > 0.1]) / len(sigma_source) * 100, len(delta_source[delta_source > 1.0]) / len(delta_source) * 100))
    
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


def main(path, index):
    """
    Main function of the plotter.
    
    Arguments:
        path (str): The path to the base folder.
        index (int): The index of the test application.
    
    Returns:
        float: The duration of the plotter.
    
    """
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'FZB/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)()
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    z_lens = [z1_lens, z2_lens]
    
    z1_source = 0.0
    z2_source = 3.0
    z_source = [z1_source, z2_source]
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    z_mean = numpy.concatenate(estimator.mean())
    z_true = test_data['photometry']['redshift']
    mag_source = test_data['photometry']['mag_i_lsst']
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure = plot_select(z_grid, z_mean, z_lens, z_source, mag_source)
    figure.savefig(os.path.join(plot_path, 'SELECT/SELECT{}.pdf'.format(index)), bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_redshift(z_grid, z_mean, z_true, z_lens, z_source, mag_source)
    figure.savefig(os.path.join(plot_path, 'REDSHIFT/REDSHIFT{}.pdf'.format(index)), bbox_inches='tight')
    pyplot.close(figure)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Figure')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            RESULT = POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])