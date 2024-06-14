import os
import time
import numpy
import argparse
from rail import core
from matplotlib import pyplot, colors

def plot_select(z0, mag0, z_grid, z_mode, z_test, mag_test):
    """
    Plot the magnitude distribution of source redshifts.
    
    Parameters:
        z_grid (numpy.ndarray): The redshift grid of redshift PDF.
        z_mode (numpy.ndarray): The redshift mode of source samples.
        z_test (numpy.ndarray): The redshifts of test application samples.
        mag_test (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    
    """
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    z_bin_size = 100
    z1 = z_grid.min()
    z2 = z_grid.max()
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    
    mag1 = 16.0
    mag2 = 26.0
    mag_bin_size = 100
    mag_bin = numpy.linspace(mag1, mag2, mag_bin_size + 1)
    
    # Plot
    width = 10
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_map, z_edge, z_edge, z_mesh = plot[0].hist2d(x=z_mode, y=mag_test, bins=[z_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[0].plot(z_grid[z_grid <= 1.5], 4 * z_grid[z_grid <= 1.5] + 18, color='black', linestyle='-', linewidth=2.0)
    
    plot[0].plot(z_grid[(z_grid >= 1.5) & (z_grid <= z0)], numpy.ones_like(z_grid[(z_grid >= 1.5) & (z_grid <= z0)]) * mag0, color='black', linestyle='-', linewidth=2.0)
    
    plot[0].plot(numpy.ones(width + 1) * z0, numpy.linspace(mag1, mag0, width + 1), color='black', linestyle='-', linewidth=2.0)
    
    plot[0].set_xlim(z1, z2)
    plot[0].set_ylim(mag1, mag2)
    
    plot[0].set_ylabel(r'$i$')
    plot[0].set_xlabel(r'$z_\mathrm{mode}$')
    
    # Plot 2
    z_map, z_edge, z_edge, z_mesh = plot[1].hist2d(x=z_test, y=mag_test, bins=[z_bin, mag_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[1].plot(z_grid[z_grid <= 1.5], 4 * z_grid[z_grid <= 1.5] + 18, color='black', linestyle='-', linewidth=2.0)
    
    plot[1].plot(z_grid[(z_grid >= 1.5) & (z_grid <= z0)], numpy.ones_like(z_grid[(z_grid >= 1.5) & (z_grid <= z0)]) * mag0, color='black', linestyle='-', linewidth=2.0)
    
    plot[1].plot(numpy.ones(width + 1) * z0, numpy.linspace(mag1, mag0, width + 1), color='black', linestyle='-', linewidth=2.0)
    
    plot[1].set_xlim(z1, z2)
    plot[1].set_ylim(mag1, mag2)
    
    plot[1].set_xlabel(r'$z_\mathrm{true}$')
    
    plot[1].set_yticklabels([])
    plot[1].get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00)
    return figure


def plot_redshift(z0, mag0, z_grid, z_mode, z_test, mag_test):
    """
    Plot the photometric redshift performance of source samples.
    
    Parameters:
        z_grid (numpy.ndarray): The redshift grid of redshift PDF.
        z_mode (numpy.ndarray): The redshift mode of source samples.
        z_test (numpy.ndarray): The redshifts of test application samples.
        mag_test (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        matplotlib.figure.Figure: The plotted figure.
    
    """
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    z_bin_size = 100
    z1 = z_grid.min()
    z2 = z_grid.max()
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    select = (mag_test < 4 * z_mode + 18) & (mag_test < mag0) & (z_mode < z0)
    print(len(select[select]))
    # Plot
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_map, z_edge, z_edge, z_mesh = plot[0].hist2d(x=z_test[select], y=z_mode[select], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[0].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[0].set_xlim(z1, z2)
    plot[0].set_ylim(z1, z2)
    
    plot[0].set_title(r'$\mathrm{Lens}$')
    plot[0].set_ylabel(r'$z_\mathrm{mode}$')
    plot[0].set_xlabel(r'$z_\mathrm{true}$')
    
    # Plot 2
    z_map, z_edge, z_edge, z_mesh = plot[1].hist2d(x=z_test, y=z_mode, bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[1].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[1].set_xlim(z1, z2)
    plot[1].set_ylim(z1, z2)
    
    plot[1].set_title(r'$\mathrm{Source}$')
    plot[1].set_xlabel(r'$z_\mathrm{true}$')
    
    plot[1].set_yticklabels([])
    plot[1].get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00)
    return figure


def main(path, index):
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA')
    plot_path = os.path.join(path, 'PLOT')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'ESTIMATE/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Redshift
    z0 = 2.0
    z1 = 0.0
    z2 = 3.0
    z_size = 300
    z_grid = numpy.linspace(z1, z2, z_size + 1)
    
    mag0 = 24.0
    mag_test = test_data()['photometry']['mag_i_lsst']
    
    z_test = test_data()['photometry']['redshift']
    z_mode = numpy.concatenate(estimator().mode(grid=z_grid))
    
    # Plot
    figure = plot_select(z0, mag0, z_grid, z_mode, z_test, mag_test)
    figure.savefig(os.path.join(plot_path, 'SELECT/SELECT{}.pdf'.format(index)), bbox_inches='tight')
    
    figure = plot_redshift(z0, mag0, z_grid, z_mode, z_test, mag_test)
    figure.savefig(os.path.join(plot_path, 'REDSHIFT/REDSHIFT{}.pdf'.format(index)), bbox_inches='tight')

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the train datasets')
    
    PATH = PARSE.parse_args().path
    INDEX = PARSE.parse_args().index
    print('Index: {}'.format(INDEX))
    
    START = time.time()
    main(PATH, INDEX)
    
    END = time.time()
    print('Time: {:.2f} minutes'.format((END - START) / 60))