import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot


def main(folder):
    '''
    Plot the distribution of the mean redshift for the lens and source galaxies.
    
    Arguments:
        folder (str): The path to the base folder containing the datasets.
    
    Returns:
        duration (float): The duration of the process in minutes.
    '''
    # Start
    start = time.time()
    
    # Path
    som_folder = os.path.join(folder, 'SOM')
    
    # Ensemble
    with h5py.File(os.path.join(som_folder, 'LENS/ENSEMBLE.hdf5'), 'r') as file:
        lens_data = file['data'][:].astype(numpy.float32)
        lens_select = file['select'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(som_folder, 'SOURCE/ENSEMBLE.hdf5'), 'r') as file:
        source_data = file['data'][:].astype(numpy.float32)
        source_select = file['select'][:].astype(numpy.float32)
    
    z1 = 0.0
    z2 = 3.0
    bin_size = 5
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    lens_mean_data = numpy.sum(lens_data * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    source_mean_data = numpy.sum(source_data * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    lens_mean_select = numpy.sum(lens_select * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    source_mean_select = numpy.sum(source_select * z_grid[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    lens_center_select = numpy.median(lens_mean_select, axis=0)
    source_center_select = numpy.median(source_mean_select, axis=0)
    
    lens_center_data = numpy.median(lens_mean_data, axis=0)
    source_center_data = numpy.median(source_mean_data, axis=0)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    size = 80
    lens_shift = 0.05 * (1 + lens_center_select)
    source_shift = 0.08 * (1 + source_center_select)
    figure, plot = pyplot.subplots(ncols=2, nrows=bin_size, figsize=(12, 15))
    
    for m in range(bin_size):
        plot[m, 0].hist(lens_mean_select[:, m], bins=size, range=(lens_center_select[m] - lens_shift[m], lens_center_select[m] + lens_shift[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].hist(lens_mean_data[:, m], bins=size, range=(lens_center_select[m] - lens_shift[m], lens_center_select[m] + lens_shift[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 0].text(lens_center_select[m] - lens_shift[m] / 2, 80, r'$\Delta \langle z \rangle = {:.3f}$'.format(lens_center_data[m] - lens_center_select[m]), fontsize=20, ha='center')
        
        plot[m, 0].set_ylim(0, 120)
        plot[m, 0].set_yticklabels([])
        plot[m, 0].set_xlim(lens_center_select[m] - lens_shift[m], lens_center_select[m] + lens_shift[m])
        
        if m == bin_size - 1: 
            plot[m, 0].set_xlabel(r'$\langle z \rangle$')
        plot[m, 0].set_ylabel(r'$\mathcal{P} \left( \langle z \rangle \right)$')
        
        plot[m, 1].hist(source_mean_select[:, m], bins=size, range=(source_center_select[m] - source_shift[m], source_center_select[m] + source_shift[m]), color='darkblue', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].hist(source_mean_data[:, m], bins=size, range=(source_center_select[m] - source_shift[m], source_center_select[m] + source_shift[m]), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-')
        
        plot[m, 1].text(source_center_select[m] - source_shift[m] / 2, 80, r'$\Delta \langle z \rangle = {:.3f}$'.format(source_center_data[m] - source_center_select[m]), fontsize=20, ha='center')
        
        plot[m, 1].set_ylim(0, 120)
        plot[m, 1].set_yticklabels([])
        plot[m, 1].set_xlim(source_center_select[m] - source_shift[m], source_center_select[m] + source_shift[m])
        
        if m == bin_size - 1:
            plot[m, 1].set_xlabel(r'$\langle z \rangle$')
        
    figure.subplots_adjust(wspace=0.0, hspace=0.2)
    figure.savefig(os.path.join(som_folder, 'FIGURE/FIGURE.png'), bbox_inches='tight', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble Figure')
    PARSE.add_argument('--folder', dest='path', type=str, help='Path to the base folder containing the datasets.')
    
    # Parse
    FOLDER = PARSE.parse_args().path
    
    # Output
    OUTPUT = main(FOLDER)
