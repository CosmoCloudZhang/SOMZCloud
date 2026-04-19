import os
import h5py
import time
import numpy
import pickle
import argparse
from matplotlib import colors
from matplotlib import pyplot


def main(tag, index, folder):
    '''
    Plot the SOM map and UMAP embedding of multiple datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all suites of datasets
        folder (str): The base folder of the figure
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    figure_folder = os.path.join(folder, 'FIGURE/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(figure_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(figure_folder, '{}/UMAP/'.format(tag)), exist_ok=True)
    
    # Plot
    os.environ['PATH'] = '/pscratch/sd/y/yhzhang/texlive/2025/bin/x86_64-linux:' + os.environ['PATH']
    pyplot.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    pyplot.rcParams['pgf.texsystem'] = 'pdflatex'
    pyplot.rcParams['text.usetex'] = True
    pyplot.rcParams['font.size'] = 25
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        application_cell_z_true = file['meta']['cell_z_true'][...]
        application_photometry = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    application_map = application_cell_z_true.reshape((cell_size1, cell_size2))
    
    # Features
    feature = numpy.stack([
        application_photometry['mag_u_lsst'], 
        application_photometry['mag_g_lsst'], 
        application_photometry['mag_r_lsst'], 
        application_photometry['mag_i_lsst'], 
        application_photometry['mag_z_lsst'], 
        application_photometry['mag_y_lsst'], 
        application_photometry['mag_u_lsst'] - application_photometry['mag_g_lsst'], 
        application_photometry['mag_u_lsst'] - application_photometry['mag_r_lsst'], 
        application_photometry['mag_u_lsst'] - application_photometry['mag_i_lsst'], 
        application_photometry['mag_u_lsst'] - application_photometry['mag_z_lsst'], 
        application_photometry['mag_u_lsst'] - application_photometry['mag_y_lsst'], 
        application_photometry['mag_g_lsst'] - application_photometry['mag_r_lsst'], 
        application_photometry['mag_g_lsst'] - application_photometry['mag_i_lsst'], 
        application_photometry['mag_g_lsst'] - application_photometry['mag_z_lsst'], 
        application_photometry['mag_g_lsst'] - application_photometry['mag_y_lsst'], 
        application_photometry['mag_r_lsst'] - application_photometry['mag_i_lsst'], 
        application_photometry['mag_r_lsst'] - application_photometry['mag_z_lsst'], 
        application_photometry['mag_r_lsst'] - application_photometry['mag_y_lsst'], 
        application_photometry['mag_i_lsst'] - application_photometry['mag_z_lsst'], 
        application_photometry['mag_i_lsst'] - application_photometry['mag_y_lsst'], 
        application_photometry['mag_z_lsst'] - application_photometry['mag_y_lsst'], 
    ], axis=1).astype(numpy.float32)
    
    # UMAP
    with open(os.path.join(dataset_folder, '{}/UMAP/MODEL.pkl'.format(tag)), 'rb') as file:
        model = pickle.load(file)
    reducer = model['reducer']
    embedding = reducer.transform(feature)
    coordinate1, coordinate2 = numpy.split(embedding, 2, axis=1)
    
    # Plot
    normalize = colors.Normalize(vmin=0.0, vmax=2.0)
    figure, plot = pyplot.subplots(nrows=2, ncols=1, figsize=(8, 12))
    figure.subplots_adjust(right=0.90, bottom=0.0, hspace=0.15, wspace=0.0)
    
    # Restriction
    mesh1 = plot[0].imshow(application_map, norm=normalize, cmap='rainbow', rasterized=True)
    plot[0].set_title(r'$\mathrm{SOM}$')
    plot[0].axis('off')
    
    colorbar1 = figure.add_axes([0.90, 0.50, 0.05, 0.35])
    colorbar1 = figure.colorbar(mesh1, cax=colorbar1, orientation='vertical')
    colorbar1.set_label(r'$z_\mathrm{true}$')
    
    # Combination
    mesh2 = plot[1].scatter(coordinate1, coordinate2, c=application_photometry['redshift_true'], norm=normalize, cmap='rainbow', s=0.01, marker='.', rasterized=True)
    plot[1].set_title(r'$\mathrm{UMAP}$')
    plot[1].axis('off')
    
    colorbar2 = figure.add_axes([0.90, 0.02, 0.05, 0.35])
    colorbar2 = figure.colorbar(mesh2, cax=colorbar2, orientation='vertical')
    colorbar2.set_label(r'$z_\mathrm{true}$')
    
    # Save
    figure.savefig(os.path.join(figure_folder, '{}/UMAP/FIGURE{}.pdf'.format(tag, index)), format='pdf', bbox_inches='tight', dpi=512)
    pyplot.close(figure)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Figure UMAP')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all suites of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all suites of datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)