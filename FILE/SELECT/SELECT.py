import os
import time
import numpy
import argparse
from rail import core
from matplotlib import pyplot, colors

def plot(z_grid, z_mode, z_test, mag_test):
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Set variables
    mag0 = 24.0
    z_bin_size = 100
    z1 = z_grid.min()
    z2 = z_grid.max()
    z_bin = numpy.linspace(z1, z2, z_bin_size + 1)
    
    # Plot
    figure, plot = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # Plot 1
    z_map, z_edge, z_edge, z_mesh = plot[0].hist2d(x=z_test[mag_test < mag0], y=z_mode[mag_test < mag0], bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[0].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[0].set_xlim(z_bin.min(), z_bin.max())
    plot[0].set_ylim(z_bin.min(), z_bin.max())
    
    plot[0].set_title(r'$\mathrm{Lens}$')
    plot[0].set_ylabel(r'$z_\mathrm{mode}$')
    plot[0].set_xlabel(r'$z_\mathrm{true}$')
    
    # Plot 2
    z_map, z_edge, z_edge, z_mesh = plot[1].hist2d(x=z_test, y=z_mode, bins=[z_bin, z_bin], norm=colors.LogNorm(vmin=1, vmax=5000), cmap='plasma')
    
    plot[1].plot(z_grid, z_grid, color='black', linestyle='--', linewidth=2.0)
    
    plot[1].set_xlim(z_bin.min(), z_bin.max())
    plot[1].set_ylim(z_bin.min(), z_bin.max())
    
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
    z1 = 0.0
    z2 = 3.0
    z_size = 300
    z_grid = numpy.linspace(z1, z2, z_size + 1)
    
    z_test = test_data()['photometry']['redshift']
    mag_test = test_data()['photometry']['mag_i_lsst']
    z_mode = numpy.concatenate(estimator().mode(grid=z_grid))
    
    # Plot
    figure = plot(z_grid, z_mode, z_test, mag_test)
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