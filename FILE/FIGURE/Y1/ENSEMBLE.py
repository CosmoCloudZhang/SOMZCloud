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
    os.makedirs(os.path.join(figure_folder, '{}/ENSEMBLE/'.format(tag)), exist_ok=True)
    
    # Bin
    bin_lens_size = 5
    bin_source_size = 5
    ensemble_size = 500000
    color_list = ['darkmagenta', 'darkblue', 'darkgreen', 'darkorange', 'darkred']
    indices = numpy.sort(numpy.random.choice(ensemble_size, ensemble_size // 25, replace=False))
    
    # Ensemble
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_lens = file['ensemble'][indices].astype(numpy.float32)
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/SOM.hdf5'.format(tag)), 'r') as file:
        som_lens = file['ensemble'][indices].astype(numpy.float32)
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_lens = file['ensemble'][indices].astype(numpy.float32)
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/FZB.hdf5'.format(tag)), 'r') as file:
        fzb_source = file['ensemble'][indices].astype(numpy.float32)
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/SOM.hdf5'.format(tag)), 'r') as file:
        som_source = file['ensemble'][indices].astype(numpy.float32)
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_source = file['ensemble'][indices].astype(numpy.float32)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # FZB
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (16, 16))
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, numpy.transpose(fzb_lens[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[0].plot(z_grid, numpy.mean(fzb_lens[:, m, :], axis=0), color = color_list[m], linewidth = 5.0, label=r'$m = {:.0f}$'.format(m + 1))
    
    plot[0].set_xlim(0.0, 2.0)
    plot[0].set_ylim(0.0, 8.0)
    plot[0].legend(loc='upper right')
    
    plot[0].set_xticklabels([])
    plot[0].get_yticklabels()[0].set_visible(False)
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, numpy.transpose(fzb_source[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[1].plot(z_grid, numpy.mean(fzb_source[:, m, :], axis=0), color = color_list[m], linewidth = 5.0)
    
    plot[1].set_xlim(0.0, 2.0)
    plot[1].set_ylim(0.0, 8.0)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\mathcal{P} \left( z \right)$')
    
    figure.subplots_adjust(hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/ENSEMBLE/FIGURE_FZB.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # Histogram
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (16, 16))
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, numpy.transpose(histogram_lens[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[0].plot(z_grid, numpy.mean(histogram_lens[:, m, :], axis=0), color = color_list[m], linewidth = 5.0, label=r'$m = {:.0f}$'.format(m + 1))
    
    plot[0].set_xlim(0.0, 2.0)
    plot[0].set_ylim(0.0, 8.0)
    plot[0].legend(loc='upper right')
    
    plot[0].set_xticklabels([])
    plot[0].get_yticklabels()[0].set_visible(False)
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, numpy.transpose(histogram_source[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[1].plot(z_grid, numpy.mean(histogram_source[:, m, :], axis=0), color = color_list[m], linewidth = 5.0)
    
    plot[1].set_xlim(0.0, 2.0)
    plot[1].set_ylim(0.0, 8.0)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\mathcal{P} \left( z \right)$')
    
    figure.subplots_adjust(hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/ENSEMBLE/FIGURE_HISTOGRAM.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
    # SOM
    figure, plot = pyplot.subplots(nrows = 2, ncols = 1, figsize = (16, 16))
    
    for m in range(bin_lens_size):
        
        plot[0].plot(z_grid, numpy.transpose(som_lens[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[0].plot(z_grid, numpy.mean(som_lens[:, m, :], axis=0), color = color_list[m], linewidth = 5.0, label=r'$m = {:.0f}$'.format(m + 1))
    
    plot[0].set_xlim(0.0, 2.0)
    plot[0].set_ylim(0.0, 8.0)
    plot[0].legend(loc='upper right')
    
    plot[0].set_xticklabels([])
    plot[0].get_yticklabels()[0].set_visible(False)
    
    for m in range(bin_source_size):
        
        plot[1].plot(z_grid, numpy.transpose(som_source[:, m, :]), color = color_list[m], linewidth = 0.04, alpha = 0.04)
        
        plot[1].plot(z_grid, numpy.mean(som_source[:, m, :], axis=0), color = color_list[m], linewidth = 5.0)
    
    plot[1].set_xlim(0.0, 2.0)
    plot[1].set_ylim(0.0, 8.0)
    
    plot[1].set_xlabel(r'$z$')
    plot[1].set_ylabel(r'$\mathcal{P} \left( z \right)$')
    
    figure.subplots_adjust(hspace=0.0)
    figure.savefig(os.path.join(figure_folder, '{}/ENSEMBLE/FIGURE_SOM.png'.format(tag)), bbox_inches='tight', format='png', dpi=512)
    
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
