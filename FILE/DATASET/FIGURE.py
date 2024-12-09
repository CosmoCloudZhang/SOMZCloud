import os
import h5py
import time
import numpy
import argparse
import multiprocessing
from matplotlib import colors, pyplot
from matplotlib.gridspec import GridSpec


def plot_figure(index, folder):
    '''
    Plot the figure of the datasets
    
    Arguments:
        index (int): The index of the dataset
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Application
    with h5py.File(os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index)), 'r') as file:
        application_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_application = application_data['redshift']
    mag_application = application_data['mag_i_lsst']
    color_application = numpy.subtract(application_data['mag_g_lsst'], application_data['mag_z_lsst'], where=(application_data['mag_g_lsst'] != 99.0) & (application_data['mag_z_lsst'] != 99.0), out=numpy.full_like(application_data['mag_i_lsst'], 99.0))
    
    # Selection
    with h5py.File(os.path.join(dataset_folder, 'SELECTION/DATA{}.hdf5'.format(index)), 'r') as file:
        selection_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_selection = selection_data['redshift']
    mag_selection = selection_data['mag_i_lsst']
    color_selection = numpy.subtract(selection_data['mag_g_lsst'], selection_data['mag_z_lsst'], where=(selection_data['mag_g_lsst'] != 99.0) & (selection_data['mag_z_lsst'] != 99.0), out=numpy.full_like(selection_data['mag_i_lsst'], 99.0))
    
    # Augmentation
    with h5py.File(os.path.join(dataset_folder, 'AUGMENTATION/DATA{}.hdf5'.format(index)), 'r') as file:
        augmentation_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_augmentation = augmentation_data['redshift']
    mag_augmentation = augmentation_data['mag_i_lsst']
    color_augmentation = numpy.subtract(augmentation_data['mag_g_lsst'], augmentation_data['mag_z_lsst'], where=(augmentation_data['mag_g_lsst'] != 99.0) & (augmentation_data['mag_z_lsst'] != 99.0), out=numpy.full_like(augmentation_data['mag_i_lsst'], 99.0))
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, 'COMBINATION/DATA{}.hdf5'.format(index)), 'r') as file:
        combination_data = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    z_combination = combination_data['redshift']
    mag_combination = combination_data['mag_i_lsst']
    color_combination = numpy.subtract(combination_data['mag_g_lsst'], combination_data['mag_z_lsst'], where=(combination_data['mag_g_lsst'] != 99.0) & (combination_data['mag_z_lsst'] != 99.0), out=numpy.full_like(combination_data['mag_i_lsst'], 99.0))
    
    # Bin
    z1 = 0.0
    z2 = 3.0
    z_size = 50
    z_bin = numpy.linspace(z1, z2, z_size + 1)
    
    mag1 = 16.0
    mag2 = 26.0
    mag_size = 50
    mag_bin = numpy.linspace(mag1, mag2, mag_size + 1)
    
    color1 = -1.5
    color2 = +5.5
    color_size = 50
    color_bin = numpy.linspace(color1, color2, color_size + 1)
    
    figure = pyplot.figure(figsize = (12, 16))
    normalize = colors.LogNorm(vmin=1, vmax=10000)
    gridspec = GridSpec(nrows=1, ncols=2, figure=figure, top=0.95, bottom=0.75, hspace=0.2, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0,:])
    
    plot.hist(z_application, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='black', label=r'$\mathrm{application}$')
    
    plot.hist(z_selection, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkred', label=r'$\mathrm{selection}$')
    
    plot.hist(z_augmentation, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkorange', label=r'$\mathrm{augmentation}$')
    
    plot.hist(z_combination, bins=z_bin, linewidth=1.0, density=True, histtype='step', color='darkblue', label=r'$\mathrm{combination}$')
    
    plot.legend()
    plot.set_xlim(z_bin.min(), z_bin.max())
    
    plot.set_xlabel(r'$z$')
    plot.set_ylabel(r'$\mathcal{P}(z)$')
    
    gridspec = GridSpec(nrows=2, ncols=2, figure=figure, top=0.70, bottom=0.15, hspace=0.0, wspace=0.0)
    
    plot = figure.add_subplot(gridspec[0, 0])
    
    plot.text(3.0, 16.5, r'$\mathrm{application}$')
    
    mesh = plot.hist2d(color_application, mag_application, bins=[color_bin, mag_bin], norm=normalize, cmap='plasma')[-1]
    
    plot.set_ylabel(r'$i$')
    plot.set_xticklabels([])
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[0, 1])
    
    plot.text(3.0, 16.5, r'$\mathrm{selection}$')
    
    mesh = plot.hist2d(color_selection, mag_selection, bins=[color_bin, mag_bin], norm=normalize, cmap='plasma')[-1]
    
    plot.set_yticklabels([])
    plot.set_xticklabels([])
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot = figure.add_subplot(gridspec[1, 0])
    
    plot.text(3.0, 16.5, r'$\mathrm{augmentation}$')
    
    mesh = plot.hist2d(color_augmentation, mag_augmentation, bins=[color_bin, mag_bin], norm=normalize, cmap='plasma')[-1]
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_ylabel(r'$i$')
    plot.set_xlabel(r'$g - z$')
    
    plot = figure.add_subplot(gridspec[1, 1])
    
    plot.text(3.0, 16.5, r'$\mathrm{combination}$')
    
    mesh = plot.hist2d(color_combination, mag_combination, bins=[color_bin, mag_bin], norm=normalize, cmap='plasma')[-1]
    
    plot.set_ylim(mag_bin.min(), mag_bin.max())
    plot.set_xlim(color_bin.min(), color_bin.max())
    
    plot.get_yticklabels()[0].set_visible(False)
    plot.get_xticklabels()[0].set_visible(False)
    
    plot.set_yticklabels([])
    plot.set_xlabel(r'$g - z$')
    
    color_bar = figure.colorbar(mesh, cax=figure.add_axes([0.15, 0.05, 0.70, 0.05]), orientation='horizontal')
    color_bar.set_label(r'$\mathrm{Counts}$')
    figure.subplots_adjust(bottom=0.15)
    
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'FIGURE'), exist_ok=True)
    
    figure.savefig(os.path.join(dataset_folder, 'FIGURE/FIGURE{}.png'.format(index)), bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    print('Index: {} Time: {:.2f} minutes'.format(index, duration))
    return duration


def main(count, number, folder):
    '''
    Plot the figures of the datasets
    
    Arguments:
        count (int): The number of the processes
        number (int): The number of the datasets
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    size = number // count
    for chunk in range(size):
        print('Chunk: {}/{}'.format(chunk + 1, size))
        
        with multiprocessing.Pool(processes=count) as pool:
            pool.starmap(plot_figure, [(index, folder) for index in range(chunk * count + 1, (chunk + 1) * count + 1)])
        
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Combination datasets')
    PARSE.add_argument('--count', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    COUNT = PARSE.parse_args().count
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(COUNT, NUMBER, FOLDER)