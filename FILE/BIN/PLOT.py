import os
import time
import numpy
import argparse
from rail import core
import multiprocessing
from matplotlib import pyplot, colors


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
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    
    slope = 4.0
    intersection = 18.0
    
    bin_size = 100
    z_bin = numpy.linspace(z_grid.min(), z_grid.max(), bin_size + 1)
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intersection)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_mesh = plot[0].hist2d(x=z_mean[select_lens], y=z_true[select_lens], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[0].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[0].set_xlim(z1_source, z2_source)
    plot[0].set_ylim(z1_source, z2_source)
    
    plot[0].set_title(r'$\mathrm{Lens}$')
    plot[0].set_xlabel(r'$z_\mathrm{phot}$')
    plot[0].set_ylabel(r'$z_\mathrm{spec}$')
    
    # Plot 2
    z_mesh = plot[1].hist2d(x=z_mean[select_source], y=z_true[select_source], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')[-1]
    
    plot[1].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[1].set_xlim(z1_source, z2_source)
    plot[1].set_ylim(z1_source, z2_source)
    
    plot[1].set_title(r'$\mathrm{Source}$')
    plot[1].set_xlabel(r'$z_\mathrm{phot}$')
    
    plot[1].set_yticklabels([])
    plot[1].get_xticklabels()[0].set_visible(False)
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    figure.subplots_adjust(bottom=0.20, wspace=0.00)
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
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    z_lens = [z1_lens, z2_lens]
    
    z1_source = 0.0
    z2_source = 3.0
    z_source = [z1_source, z2_source]
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    z_mean = numpy.concatenate(estimator().mean())
    z_true = test_data()['photometry']['redshift']
    mag_source = test_data()['photometry']['mag_i_lsst']
    
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
    PARSE = argparse.ArgumentParser(description='FZB Plotter')
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
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])