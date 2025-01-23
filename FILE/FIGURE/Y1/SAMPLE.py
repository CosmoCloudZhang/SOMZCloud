import os
import time
import numpy
import argparse
from rail import core
from matplotlib import pyplot, colors


def main(tag, index, folder):
    '''
    Plot the figures of the sample selection
    
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
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Magnitude
    magnitude1 = 15.0
    magnitude2 = 26.0
    magnitude_grid = numpy.linspace(magnitude1, magnitude2, grid_size + 1)
    
    # Load 
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Load
    application_name = os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    magnitude = application_dataset['photometry']['mag_i_lsst']
    del application_dataset
    
    estimate_name = os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Set variables
    slope = 4.0
    intersection = 18.0
    z_lens_grid = numpy.linspace(z1_lens, z2_lens, grid_size + 1)
    z_source_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Plot
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
    z_mesh =plot.hist2d(x=z_phot, y=magnitude, bins=[z_source_grid, magnitude_grid], norm=normalize, cmap='plasma')[-1]
    
    plot.plot(z_lens_grid, slope * z_lens_grid + intersection, color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(grid_size) * z1_lens, numpy.linspace(magnitude1, slope * z1_lens + intersection, grid_size), color='black', linestyle='--', linewidth=2.0)
    
    plot.plot(numpy.ones(grid_size) * z2_lens, numpy.linspace(magnitude1, slope * z2_lens + intersection, grid_size), color='black', linestyle='--', linewidth=2.0)
    
    plot.set_xlim(z1_source, z2_source)
    plot.set_ylim(magnitude1, magnitude2)
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$z_\mathrm{phot}$')
    
    # Color bar
    color_bar = figure.colorbar(z_mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    figure.subplots_adjust(bottom=0.20, wspace=0.00, hspace=0.00)
    color_bar.set_label(r'$\mathrm{Counts}$')
    
    # Return figure
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SAMPLE/'.format(tag)), exist_ok=True)
    
    figure.savefig(os.path.join(figure_folder, '{}/SAMPLE/FIGURE{}.png'.format(tag, index)), format='png', bbox_inches='tight', dpi=512)
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