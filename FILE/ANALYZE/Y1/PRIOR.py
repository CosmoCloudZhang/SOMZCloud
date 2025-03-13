import os
import h5py
import time
import numpy
import argparse
from matplotlib import pyplot
from matplotlib import colors


def plot_prior(scale, correlation):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        scale (numpy.ndarray): The standard deviation of the prior
        correlation (numpy.ndarray): The correlation matrix of the prior
    '''
    figure, plot = pyplot.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
    
    norm = colors.Normalize(vmin = -1.0, vmax = +1.0)
    image = plot.imshow(correlation, norm = norm, cmap = 'coolwarm', origin = 'upper')
    
    for (i, j), value in numpy.ndenumerate(correlation):
        if i == j:
            plot.text(j, i, r'$\sigma_{\mu}^{' + r'{:.0f}'.format(i + 1) + r'} = ' + r'{:.3f}$'.format(scale[i]), va='center', ha='center', color='black', fontsize = 20)
        else:
            plot.text(j, i, r'${:.3f}$'.format(value), va='center', ha='center', color='black', fontsize = 20)
    
    plot.axis('off')
    cax = figure.add_axes([0.15, 0.05, 0.72, 0.05])
    figure.colorbar(cax = cax, mappable = image, orientation = 'horizontal', label = r'$\mathcal{R} \: [\Delta_{\mu}^{i}, \Delta_{\mu}^{j}]$')
    
    figure.subplots_adjust(wspace = 0.02, hspace = 0.02)
    return figure


def main(tag, folder):
    '''
    Plot the prior correlation matrix
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    
    Returns:
        figure (matplotlib.figure.Figure): The figure
    '''
    start = time.time()
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
    for label in label_list:
        
        os.makedirs(os.path.join(analyze_folder, '{}/PRIOR/'.format(tag)), exist_ok=True)
        os.makedirs(os.path.join(analyze_folder, '{}/PRIOR/{}'.format(tag, label)), exist_ok=True)
        
        # Info
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
            som_scale_lens = file['lens']['scale'][...]
            som_scale_source = file['source']['scale'][...]
            
            som_correlation_lens = file['lens']['correlation'][...]
            som_correlation_source = file['source']['correlation'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
            model_scale_lens = file['lens']['scale'][...]
            model_scale_source = file['source']['scale'][...]
            
            model_correlation_lens = file['lens']['correlation'][...]
            model_correlation_source = file['source']['correlation'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
            product_scale_lens = file['lens']['scale'][...]
            product_scale_source = file['source']['scale'][...]
            
            product_correlation_lens = file['lens']['correlation'][...]
            product_correlation_source = file['source']['correlation'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
            fiducial_scale_lens = file['lens']['scale'][...]
            fiducial_scale_source = file['source']['scale'][...]
            
            fiducial_correlation_lens = file['lens']['correlation'][...]
            fiducial_correlation_source = file['source']['correlation'][...]
        
        with h5py.File(os.path.join(analyze_folder, '{}/INFO/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
            histogram_scale_lens = file['lens']['scale'][...]
            histogram_scale_source = file['source']['scale'][...]
            
            histogram_correlation_lens = file['lens']['correlation'][...]
            histogram_correlation_source = file['source']['correlation'][...]
        
        
        # Configuration
        os.environ['PATH'] = '/global/homes/y/yhzhang/opt/texlive/bin/x86_64-linux:' + os.environ['PATH']
        pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
        pyplot.rcParams['text.usetex'] = True
        pyplot.rcParams['font.size'] = 20
        
        # Plot
        figure = plot_prior(som_scale_lens, som_correlation_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_SOM_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(som_scale_source, som_correlation_source)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_SOM_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(model_scale_lens, model_correlation_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_MODEL_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(model_scale_source, model_correlation_source)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_MODEL_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(product_scale_lens, product_correlation_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_PRODUCT_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(product_scale_source, product_correlation_source)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_PRODUCT_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(fiducial_scale_lens, fiducial_correlation_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_FIDUCIAL_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(fiducial_scale_source, fiducial_correlation_source)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_FIDUCIAL_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(histogram_scale_lens, histogram_correlation_lens)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_HISTOGRAM_LENS.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        figure = plot_prior(histogram_scale_source, histogram_correlation_source)
        figure.savefig(os.path.join(analyze_folder, '{}/PRIOR/{}/FIGURE_HISTOGRAM_SOURCE.pdf'.format(tag, label)), format='pdf', bbox_inches='tight')
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
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the figure')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)