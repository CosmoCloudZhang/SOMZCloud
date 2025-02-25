import os
import time
import h5py
import numpy
import scipy
import argparse
from rail import core
from sklearn import cluster


def main(tag, index, folder):
    '''
    Model summarization of the source samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarization_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(os.path.join(summarization_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarization_folder, '{}/SOURCE/SOURCE{}'.format(tag, index)), exist_ok=True)
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    mesh_size = 3000
    z_mesh = numpy.linspace(z1, z2, mesh_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_cell_id = file['meta']['cell_id'][...]
        application_sigma = file['morphology']['sigma'][...]
        application_size = len(file['photometry']['redshift'][...])
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        application_z_phot = file['z_phot'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_source = file['select'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_cell_id = file['meta']['cell_id'][...]
        combination_redshift = file['photometry']['redshift'][...]
    combination_size = len(combination_redshift)
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
        combination_z_phot = file['z_phot'][...]
    
    # Size
    data_size = 100
    bin_source_size = len(bin_source) - 1
    data_source = numpy.zeros((bin_source_size, data_size, grid_size + 1))
    
    # Cluster
    som_model = model['som']
    cluster_size = model['n_rows'] * model['n_columns']
    
    som_model.cluster(cluster.AgglomerativeClustering(n_clusters=cluster_size, linkage='complete'))
    cluster_id = som_model.clusters.flatten()
    
    # Chunk
    chunk_size = 10000
    estimator = h5py.File(os.path.join(model_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    # Loop
    for m in range(bin_source_size):
        # Select
        select = select_source[m, :] 
        select_size = numpy.sum(select)
        select_indices = numpy.arange(application_size)[select]
        
        # Application
        application_sigma_select = application_sigma[select]
        application_z_phot_select = application_z_phot[select]
        application_cell_id_select = application_cell_id[select]
        
        # Estimate
        pdf = numpy.zeros((data_size, grid_size + 1))
        data_weight = numpy.ones((data_size, select_size))
        application_weight_data = numpy.ones((data_size, select_size))
        
        for k in range(data_size):
            application_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            data_weight[k, :] = numpy.bincount(application_indices, minlength=select_size)
            
            # Application
            application_sigma_data = application_sigma_select[application_indices]
            application_z_phot_data = application_z_phot_select[application_indices]
            application_cell_id_data = application_cell_id_select[application_indices]
            
            application_cluster_id_data = cluster_id[application_cell_id_data]
            application_cluster_count_data = numpy.bincount(application_cluster_id_data, weights=1 / numpy.square(application_sigma_data), minlength=cluster_size)
            
            application_cluster_z_phot_data = numpy.bincount(application_cluster_id_data, weights=application_z_phot_data / numpy.square(application_sigma_data), minlength=cluster_size)
            application_cluster_z_phot_data = numpy.divide(application_cluster_z_phot_data, application_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=application_cluster_count_data > 0)
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(combination_size), size=combination_size, replace=True)
            
            combination_z_phot_data = combination_z_phot[combination_indices]
            combination_z_spec_data = combination_redshift[combination_indices]
            
            combination_cell_id_data = combination_cell_id[combination_indices]
            combination_cluster_id_data = cluster_id[combination_cell_id_data]
            combination_cluster_count_data = numpy.bincount(combination_cluster_id_data, minlength=cluster_size)
            
            combination_cluster_z_phot_data = numpy.bincount(combination_cluster_id_data, weights=combination_z_phot_data, minlength=cluster_size)
            combination_cluster_z_phot_data = numpy.divide(combination_cluster_z_phot_data, combination_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_data > 0)
            
            combination_cluster_z_spec_data = numpy.bincount(combination_cluster_id_data, weights=combination_z_spec_data, minlength=cluster_size)
            combination_cluster_z_spec_data = numpy.divide(combination_cluster_z_spec_data, combination_cluster_count_data, out=numpy.zeros(cluster_size, dtype=numpy.float32), where=combination_cluster_count_data > 0)
            
            # Filter
            filter_data = (application_cluster_count_data > 0) & (combination_cluster_count_data > 0)
            cluster_mean_delta_data = application_cluster_z_phot_data - combination_cluster_z_spec_data
            sigma_data = 1.4826 * numpy.median(numpy.abs(cluster_mean_delta_data[filter_data] - numpy.median(cluster_mean_delta_data[filter_data])))
            filter_data = filter_data & (numpy.abs(cluster_mean_delta_data) - 5 * sigma_data < 0) & (numpy.abs(combination_cluster_z_phot_data - combination_cluster_z_spec_data) < 0.02)
            
            # Weight
            cluster_weight_data = numpy.array(filter_data, dtype=numpy.float32)
            application_weight_data[k, :] = cluster_weight_data[application_cluster_id_data] / numpy.square(application_sigma_data)
        
        # Loop
        for n in range(select_size // chunk_size + 1):
            # PDF
            begin = n * chunk_size
            end = min((n + 1) * chunk_size, application_size)
            z_pdf = estimator['data']['yvals'][select_indices[begin: end]].astype(numpy.float32)
            
            # Histogram
            pdf = pdf + numpy.sum(z_pdf[numpy.newaxis, :, :] * data_weight[:, begin: end][:, :, numpy.newaxis], axis=1)
        
        # Random
        for n in range(data_size):
            pdf_data = numpy.maximum(scipy.interpolate.CubicSpline(x=z_grid, y=pdf[n, :], bc_type='natural', extrapolate=False)(z_mesh), 0)
            z_data = numpy.random.choice(z_mesh, size=select_size, replace=True, p=pdf_data / numpy.sum(pdf_data))
            
            histogram = numpy.histogram(z_data, bins=z_bin, range=(z1, z2), weights=application_weight_data[n, :], density=True)[0]
            data_source[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(summarization_folder, '{}/SOURCE/SOURCE{}/MODEL.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('data', data=data_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Model')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
