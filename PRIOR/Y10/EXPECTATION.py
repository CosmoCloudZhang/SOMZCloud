import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def plot_expectation(sigma, rho_mu):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        sigma (numpy.ndarray): The standard deviation of the prior
        correlation (numpy.ndarray): The correlation matrix of the prior
    '''
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (3 * len(sigma), 3 * len(sigma)))
    
    norm = colors.Normalize(vmin = -1.0, vmax = +1.0)
    image = plot.imshow(rho_mu, norm = norm, cmap = 'coolwarm', origin = 'upper')
    
    for (i, j), value in numpy.ndenumerate(rho_mu):
        if i == j:
            plot.text(j, i, r'$\sigma_\mu = ' + r'{:.3f}$'.format(sigma[i]), va='center', ha='center', color='black', fontsize = 30)
        else:
            plot.text(j, i, r'${:.3f}$'.format(value), va='center', ha='center', color='black', fontsize = 30)
    
    for i in range(len(sigma)):
        plot.text(i, -1, r'$\mathrm{Bin} \, ' + r'{:.0f}$'.format(i + 1), va='center', ha='center', color='black', fontsize = 30)
    
    for j in range(len(sigma)):
        plot.text(-1, j, r'$\mathrm{Bin} \, ' + r'{:.0f}$'.format(j + 1), va='center', ha='center', color='black', fontsize = 30)
    
    plot.axis('off')
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{R}_\mu$')
    
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    return figure


def main(tag, name, label, folder):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    start = time.time()
    print('name: {}, Label: {}'.format(name, label))
    
    # Path
    prior_folder = os.path.join(folder, 'PRIOR/')
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    os.makedirs(os.path.join(prior_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/EXPECTATION/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/EXPECTATION/{}/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(prior_folder, '{}/EXPECTATION/{}/{}/'.format(tag, name, label)), exist_ok=True)
    
    # Info
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        
        sigma_lens = file['lens']['sigma_mu'][...]
        sigma_source = file['source']['sigma_mu'][...]
        
        rho_mu_lens = file['lens']['rho_mu'][...]
        rho_mu_source = file['source']['rho_mu'][...]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/Bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot Expectation
    figure = plot_expectation(sigma_lens, rho_mu_lens)
    figure.savefig(os.path.join(prior_folder, '{}/EXPECTATION/{}/{}/FIGURE_LENS.pdf'.format(tag, name, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_expectation(sigma_source, rho_mu_source)
    figure.savefig(os.path.join(prior_folder, '{}/EXPECTATION/{}/{}/FIGURE_SOURCE.pdf'.format(tag, name, label)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Prior Expectation')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, FOLDER)