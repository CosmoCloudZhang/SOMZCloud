import os
import h5py
import time
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
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load 
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        mu = file['mu'][...]
        eta = file['eta'][...]
        sigma = file['sigma'][...]
    bin_size = 100
    
    mu1 = 0
    mu2 = 40
    mu_edge = numpy.linspace(mu1, mu2, bin_size + 1)
    
    eta1 = 0
    eta2 = 4
    eta_edge = numpy.linspace(eta1, eta2, bin_size + 1)
    
    map = numpy.zeros((bin_size, bin_size))
    for m in range(bin_size):
        for n in range(bin_size):
            select = (mu_edge[m] <= mu) & (mu < mu_edge[m + 1]) 
            select = select & (eta_edge[n] <= eta) & (eta < eta_edge[n + 1])
            
            if numpy.sum(select) > 0:
                map[n, m] = numpy.mean(sigma[select])
            else:
                map[n, m] = numpy.nan
    
    # Plot
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    figure, plot = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    image = plot.imshow(map, origin='lower', cmap='viridis', extent=[mu1, mu2, eta1, eta2], aspect='auto')
    
    plot.plot(mu_edge, numpy.ones(bin_size + 1) * 0.1, color='black', linestyle='--', linewidth=2.5)
    
    plot.plot(numpy.ones(bin_size + 1) * 10, eta_edge, color='black', linestyle='--', linewidth=2.5)
    
    plot.set_xlim(mu1, mu2)
    plot.set_ylim(eta1, eta2)
    
    plot.set_xlabel(r'$\mu = \frac{S}{N}$')
    plot.set_ylabel(r'$\eta = \frac{R^2}{R^2_\mathrm{PSF}}$')
    
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.85, 0.15, 0.05, 0.70]),  orientation='vertical')
    color_bar.set_label(r'$\sigma_\gamma$')
    
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/SIGMA/'.format(tag)), exist_ok=True)
    
    figure.subplots_adjust(right=0.80)
    figure.savefig(os.path.join(figure_folder, '{}/SIGMA/FIGURE.pdf'.format(tag)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
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