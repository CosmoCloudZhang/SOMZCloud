import os
import time
import h5py
import numpy
import argparse
from matplotlib import pyplot


def main(path):
    # Data
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    plot_path = os.path.join(path, 'PLOT/')
    os.makedirs(os.path.join(plot_path, 'ENSEMBLE/'), exist_ok=True)
    
    # Ensemble
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/LENS/ENSEMBLE.hdf5'), 'r') as file:
        lens_ensemble_data = file['data'][:].astype(numpy.float32)
        lens_ensemble_sample = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/ENSEMBLE.hdf5'), 'r') as file:
        source_ensemble_data = file['data'][:].astype(numpy.float32)
        source_ensemble_sample = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/LENS/FZB_ENSEMBLE.hdf5'), 'r') as file:
        lens_fzb_ensemble_data = file['data'][:].astype(numpy.float32)
        lens_fzb_ensemble_sample = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/FZB_ENSEMBLE.hdf5'), 'r') as file:
        source_fzb_ensemble_data = file['data'][:].astype(numpy.float32)
        source_fzb_ensemble_sample = file['sample'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/LENS/SOM_ENSEMBLE.hdf5'), 'r') as file:
        som_lens_ensemble_data = file['data'][:].astype(numpy.float32)
        som_lens_ensemble_sample = file['sample'][:].astype(numpy.float32)
    
    z1 = 0.0
    z2 = 3.0
    size = 100
    bin_size = 5
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_data = numpy.linspace(z1 + z_delta / 2, z2 - z_delta / 2, grid_size)
    
    lens_mean_data = numpy.sum(lens_ensemble_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    lens_mean_sample = numpy.sum(lens_ensemble_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    source_mean_data = numpy.sum(source_ensemble_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    source_mean_sample = numpy.sum(source_ensemble_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    lens_fzb_mean_data = numpy.sum(lens_fzb_ensemble_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    lens_fzb_mean_sample = numpy.sum(lens_fzb_ensemble_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    source_fzb_mean_data = numpy.sum(source_fzb_ensemble_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    source_fzb_mean_sample = numpy.sum(source_fzb_ensemble_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    som_lens_mean_data = numpy.sum(som_lens_ensemble_data * z_data[numpy.newaxis, :], axis=1) * z_delta
    som_lens_mean_sample = numpy.sum(som_lens_ensemble_sample * z_data[numpy.newaxis, numpy.newaxis, :], axis=2) * z_delta
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot
    shift1 = -0.025
    shift2 = +0.025
    figure, plot = pyplot.subplots(ncols=2, nrows=bin_size, figsize=(12, 15))
    
    for m in range(bin_size):
        plot[m, 0].hist(lens_mean_sample[:, m] - lens_mean_data[m], bins=size, range=(shift1, shift2), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-', label=r'$\mathrm{Fiducial}$')
        
        plot[m, 0].hist(lens_fzb_mean_sample[:, m] - lens_fzb_mean_data[m], bins=size, range=(shift1, shift2), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-', label=r'$\mathrm{FZB}$')
        
        plot[m, 0].hist(som_lens_mean_sample[:, m] - som_lens_mean_data[m], bins=size, range=(shift1, shift2), color='darkorange', density=True, histtype='step', linewidth=2.0, linestyle='-', label=r'$\mathrm{SOM}$')
        
        plot[m, 0].set_xlim(shift1, shift2)
        plot[m, 0].legend(loc='upper right')
        
        plot[m, 1].hist(source_mean_sample[:, m] - source_mean_data[m], bins=size, range=(shift1, shift2), color='black', density=True, histtype='step', linewidth=2.0, linestyle='-', label=r'$\mathrm{Fiducial}$')
        
        plot[m, 1].hist(source_fzb_mean_sample[:, m] - source_fzb_mean_data[m], bins=size, range=(shift1, shift2), color='darkred', density=True, histtype='step', linewidth=2.0, linestyle='-', label=r'$\mathrm{FZB}$')
        
        plot[m, 1].set_yticklabels([])
        plot[m, 1].set_xlim(shift1, shift2)
        plot[m, 1].legend(loc='upper right')
        
        if m != bin_size - 1:
            plot[m, 0].set_xticklabels([])
            plot[m, 1].set_xticklabels([])
    
    figure.subplots_adjust(wspace=0.0, hspace=0.0)
    figure.savefig(os.path.join(plot_path, 'ENSEMBLE/ENSEMBLE.pdf'), bbox_inches='tight')
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    
    PATH = PARSE.parse_args().path
    RESULT = main(PATH)