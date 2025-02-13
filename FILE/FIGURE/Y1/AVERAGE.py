import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot

def main(tag, folder):
    '''
    Plot the shear noise as function of the galaxy sizes and brightnesses
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/AVERAGE/'.format(tag)), exist_ok=True)
    
    # Ensemble
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_lens = file['average'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['average'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['average'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_source = file['average'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/SOM.hdf5'.format(tag)), 'r') as file:
        som_source = file['average'][...]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_source = file['average'][...]
    
    # Bin
    bin_lens_size = 5
    bin_source_size = 5
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    figure, plot = pyplot.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 16))
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, histogram_lens[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[0].plot(z_grid, fzb_lens[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[0].plot(z_grid, som_lens[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
    
    plot[0].set_ylim(0, 8)
    plot[0].set_xlim(0.0, 2.0)
    
    plot[0].set_xticklabels([])
    plot[0].get_yticklabels()[0].set_visible(False)
    plot[0].set_ylabel(r'$\mathcal{P} \left( z \right)$')
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, histogram_source[m, :], color='darkblue', linewidth=2.0, linestyle='-')
        
        plot[1].plot(z_grid, fzb_source[m, :], color='darkorange', linewidth=2.0, linestyle='-')
        
        plot[1].plot(z_grid, som_source[m, :], color='darkgreen', linewidth=2.0, linestyle='-')
    
    plot[1].set_ylim(0, 8)
    plot[1].set_xlim(0.0, 2.0)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\mathcal{P} \left( z \right)$')
    
    figure.subplots_adjust(hspace=0.05)
    figure.savefig(os.path.join(figure_folder, '{}/AVERAGE/FIGURE.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Sigma')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)
