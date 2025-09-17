import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def plot_expectation(sigma, correlation):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        sigma (numpy.ndarray): The standard deviation of the prior
        correlation (numpy.ndarray): The correlation matrix of the prior
    '''
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (3 * len(sigma), 3 * len(sigma)))
    
    norm = colors.Normalize(vmin = -1.0, vmax = +1.0)
    image = plot.imshow(correlation, norm = norm, cmap = 'coolwarm', origin = 'upper')
    
    for (i, j), value in numpy.ndenumerate(correlation):
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


def plot_deviation(sigma, correlation):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        sigma (numpy.ndarray): The standard deviation of the prior
        correlation (numpy.ndarray): The correlation matrix of the prior
    '''
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (3 * len(sigma), 3 * len(sigma)))
    
    norm = colors.Normalize(vmin = -1.0, vmax = +1.0)
    image = plot.imshow(correlation, norm = norm, cmap = 'coolwarm', origin = 'upper')
    
    for (i, j), value in numpy.ndenumerate(correlation):
        if i == j:
            plot.text(j, i, r'$\sigma_\eta = ' + r'{:.3f}$'.format(sigma[i]), va='center', ha='center', color='black', fontsize = 30)
        else:
            plot.text(j, i, r'${:.3f}$'.format(value), va='center', ha='center', color='black', fontsize = 30)
    
    for i in range(len(sigma)):
        plot.text(i, -1, r'$\mathrm{Bin} \, ' + r'{:.0f}$'.format(i + 1), va='center', ha='center', color='black', fontsize = 30)
    
    for j in range(len(sigma)):
        plot.text(-1, j, r'$\mathrm{Bin} \, ' + r'{:.0f}$'.format(j + 1), va='center', ha='center', color='black', fontsize = 30)
    
    plot.axis('off')
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{R}_\eta$')
    
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    return figure


def main(tag, rank, label, folder):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        tag (str): The tag of the configuration
        rank (str): The rank of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    start = time.time()
    print('Rank: {}, Label: {}'.format(rank, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/PRIOR/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/PRIOR/{}'.format(tag, label)), exist_ok=True)
    
    # Info
    with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/{}_{}.hdf5'.format(tag, rank, label)), 'r') as file:
        
        scatter_lens = file['lens']['scatter'][...]
        scatter_source = file['source']['scatter'][...]
        
        variation_lens = file['lens']['variation'][...]
        variation_source = file['source']['variation'][...]
        
        correlation_deviation_lens = file['lens']['correlation_deviation'][...]
        correlation_deviation_source = file['source']['correlation_deviation'][...]
        
        correlation_expectation_lens = file['lens']['correlation_expectation'][...]
        correlation_expectation_source = file['source']['correlation_expectation'][...]
    
    # Configuration
    os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/Bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 20
    
    # Plot Expectation
    figure = plot_expectation(scatter_lens, correlation_expectation_lens)
    figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_{}_EXPECTATION_LENS.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_expectation(scatter_source, correlation_expectation_source)
    figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_{}_EXPECTATION_SOURCE.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Plot Deviation
    figure = plot_deviation(variation_lens, correlation_deviation_lens)
    figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_{}_DEVIATION_LENS.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    figure = plot_deviation(variation_source, correlation_deviation_source)
    figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_{}_DEVIATION_SOURCE.pdf'.format(tag, label, rank)), format='pdf', bbox_inches='tight')
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Prior')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--rank', type=str, required=True, help='The rank of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    RANK = PARSE.parse_args().rank
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, RANK, LABEL, FOLDER)