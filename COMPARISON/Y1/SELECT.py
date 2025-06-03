import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Define the lens and source selection
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the process
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    comparison_folder = os.path.join(folder, 'COMPARISON/')
    os.makedirs(os.path.join(comparison_folder, '{}/SELECT/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(comparison_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(comparison_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(comparison_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(comparison_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.05
    z2_source = 2.95
    
    z1 = 0.0
    z2 = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    mesh_size = 3000
    z_delta = (z2 - z1) / mesh_size
    z_mesh = numpy.linspace(z1, z2, mesh_size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_magnitude = file['photometry']['mag_i_lsst'][...]
        application_redshift_true = file['photometry']['redshift_true'][...]
    application_size = len(application_magnitude)
    
    # Estimate
    chunk_size = 100000
    estimator = h5py.File(os.path.join(comparison_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    z_loss = numpy.zeros(application_size, dtype=numpy.float32)
    z_phot = numpy.zeros(application_size, dtype=numpy.float32)
    z_quantile = numpy.zeros(application_size, dtype=numpy.float32)
    
    for m in range(application_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, application_size)
        
        if begin < stop:
            z_pdf = numpy.maximum(scipy.interpolate.CubicSpline(x=z_grid, y=estimator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate=False)(z_mesh), 0.0)
            z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
            z_phot[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
            
            z_cdf = numpy.cumsum(z_pdf, axis=1) * z_delta
            z_quantile[begin: stop] = z_cdf[numpy.arange(stop - begin), numpy.round((application_redshift_true[begin: stop] - z1) / z_delta).astype(numpy.int32)]
            z_loss[begin: stop] = numpy.sqrt(numpy.sum(numpy.square(z_cdf - numpy.array(z_mesh[numpy.newaxis, :] >= application_redshift_true[begin: stop, numpy.newaxis], dtype=numpy.float32)), axis=1) * z_delta / (z2 - z1))
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = (z1_source <= z_phot) & (z_phot < z2_source)
    select_lens = (z1_lens <= z_phot) & (z_phot < z2_lens) & (application_magnitude < slope * z_phot + intercept)
    
    # Bin
    lens_size = 5
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    bin_source = numpy.quantile(z_phot[select_source], numpy.linspace(0, 1, source_size + 1))
    
    bin_source[0] = z1_source
    bin_source[-1] = z2_source
    
    # Point
    z_phot_lens = z_phot[select_lens]
    z_true_lens = application_redshift_true[select_lens]
    
    z_phot_source = z_phot[select_source]
    z_true_source = application_redshift_true[select_source]
    
    # Loss
    z_loss_lens = z_loss[select_lens]
    z_loss_source = z_loss[select_source]
    
    # Quantile
    z_quantile_lens = z_quantile[select_lens]
    z_quantile_source = z_quantile[select_source]
    
    # Histogram
    histogram_lens_size = 10
    histogram_bin_lens = numpy.linspace(0, 1, histogram_lens_size + 1)
    
    histogram_source_size = 20
    histogram_bin_source = numpy.linspace(0, 1, histogram_source_size + 1)
    
    # Metric Lens
    z1_average_lens = 0.0
    z2_average_lens = 1.5
    average_lens_size = 5
    z_average_lens = numpy.linspace(z1_average_lens, z2_average_lens, average_lens_size + 1)
    
    rate_lens = numpy.zeros(average_lens_size)
    delta_lens = numpy.zeros(average_lens_size)
    sigma_lens = numpy.zeros(average_lens_size)
    fraction_lens = numpy.zeros(average_lens_size)
    
    score_lens = numpy.zeros(average_lens_size)
    divergence_lens = numpy.zeros(average_lens_size)
    histogram_lens = numpy.zeros((average_lens_size, histogram_lens_size))
    
    for m in range(average_lens_size):
        select_lens_average = (z_average_lens[m] <= z_true_lens) & (z_true_lens < z_average_lens[m + 1])
        if numpy.sum(select_lens_average) > 0:
            z_phot_lens_select = z_phot_lens[select_lens_average]
            z_true_lens_select = z_true_lens[select_lens_average]
            delta_lens_select = numpy.abs(z_phot_lens_select - z_true_lens_select) / (1 + z_true_lens_select)
            
            delta_lens[m] = numpy.median(delta_lens_select)
            sigma_lens[m] = scipy.stats.median_abs_deviation(delta_lens_select, scale='normal')
            fraction_lens[m] = numpy.sum(delta_lens_select > 0.15) / numpy.sum(select_lens_average)
            rate_lens[m] = numpy.sum(numpy.abs(z_phot_lens_select - z_true_lens_select) > 1.0) / numpy.sum(select_lens_average)
            
            z_loss_lens_select = z_loss_lens[select_lens_average]
            score_lens[m] = numpy.median(z_loss_lens_select)
            
            z_quantile_lens_select = z_quantile_lens[select_lens_average]
            histogram_lens[m, :] = numpy.histogram(z_quantile_lens_select, bins=histogram_bin_lens, range=(0, 1), density=True)[0]
            divergence_lens[m] = numpy.sqrt(numpy.sum(numpy.square(histogram_lens[m, :] - numpy.ones(histogram_lens_size))) / histogram_lens_size)
        else:
            rate_lens[m] = numpy.nan
            delta_lens[m] = numpy.nan
            sigma_lens[m] = numpy.nan
            fraction_lens[m] = numpy.nan
            
            score_lens[m] = numpy.nan
            divergence_lens[m] = numpy.nan
            histogram_lens[m, :] = numpy.nan
    
    # Metric Source
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_source_size = 10
    z_average_source = numpy.linspace(z1_average_source, z2_average_source, average_source_size + 1)
    
    rate_source = numpy.zeros(average_source_size)
    delta_source = numpy.zeros(average_source_size)
    sigma_source = numpy.zeros(average_source_size)
    fraction_source = numpy.zeros(average_source_size)
    
    score_source = numpy.zeros(average_source_size)
    divergence_source = numpy.zeros(average_source_size)
    histogram_source = numpy.zeros((average_source_size, histogram_source_size))
    
    for m in range(average_source_size):
        select_source_average = (z_average_source[m] <= z_true_source) & (z_true_source < z_average_source[m + 1])
        if numpy.sum(select_source_average) > 0:
            z_phot_source_select = z_phot_source[select_source_average]
            z_true_source_select = z_true_source[select_source_average]
            delta_source_select = numpy.abs(z_phot_source_select - z_true_source_select) / (1 + z_true_source_select)
            
            delta_source[m] = numpy.median(delta_source_select)
            sigma_source[m] = scipy.stats.median_abs_deviation(delta_source_select, scale='normal')
            fraction_source[m] = numpy.sum(delta_source_select > 0.15) / numpy.sum(select_source_average)
            rate_source[m] = numpy.sum(numpy.abs(z_phot_source_select - z_true_source_select) > 1.0) / numpy.sum(select_source_average)
            
            z_loss_source_select = z_loss_source[select_source_average]
            score_source[m] = numpy.median(z_loss_source_select)
            
            z_quantile_source_select = z_quantile_source[select_source_average]
            histogram_source[m, :] = numpy.histogram(z_quantile_source_select, bins=histogram_bin_source, range=(0, 1), density=True)[0]
            divergence_source[m] = numpy.sqrt(numpy.sum(numpy.square(histogram_source[m, :] - numpy.ones(histogram_source_size))) / histogram_source_size)
        else:
            rate_source[m] = numpy.nan
            delta_source[m] = numpy.nan
            sigma_source[m] = numpy.nan
            fraction_source[m] = numpy.nan
            
            score_source[m] = numpy.nan
            divergence_source[m] = numpy.nan
            histogram_source[m, :] = numpy.nan
    
    # Save
    with h5py.File(os.path.join(comparison_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('z_phot', data=z_phot, dtype=numpy.float32)
        file.create_dataset('z_loss', data=z_loss, dtype=numpy.float32)
        file.create_dataset('z_quantile', data=z_quantile, dtype=numpy.float32)
        
        # Lens
        file.create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file.create_dataset('select_lens', data=select_lens, dtype=bool)
        
        file.create_dataset('rate_lens', data=rate_lens, dtype=numpy.float32)
        file.create_dataset('delta_lens', data=delta_lens, dtype=numpy.float32)
        file.create_dataset('sigma_lens', data=sigma_lens, dtype=numpy.float32)
        file.create_dataset('fraction_lens', data=fraction_lens, dtype=numpy.float32)
        
        file.create_dataset('score_lens', data=score_lens, dtype=numpy.float32)
        file.create_dataset('histogram_lens', data=histogram_lens, dtype=numpy.float32)
        file.create_dataset('divergence_lens', data=divergence_lens, dtype=numpy.float32)   
        
        # Source
        file.create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file.create_dataset('select_source', data=select_source, dtype=bool)
        
        file.create_dataset('rate_source', data=rate_source, dtype=numpy.float32)
        file.create_dataset('delta_source', data=delta_source, dtype=numpy.float32)
        file.create_dataset('sigma_source', data=sigma_source, dtype=numpy.float32)
        file.create_dataset('fraction_source', data=fraction_source, dtype=numpy.float32)
        
        file.create_dataset('score_source', data=score_source, dtype=numpy.float32)
        file.create_dataset('histogram_source', data=histogram_source, dtype=numpy.float32)
        file.create_dataset('divergence_source', data=divergence_source, dtype=numpy.float32)
    
    # Lens bin
    select_lens_bin = numpy.ones((lens_size, application_size), dtype=bool)
    for m in range(len(bin_lens) - 1):
        select_lens_bin[m, :] = select_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
    
    with h5py.File(os.path.join(comparison_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('select', data=select_lens_bin, dtype=bool)
    
    # Source bin
    select_source_bin = numpy.ones((source_size, application_size), dtype=bool)
    for m in range(len(bin_source) - 1):
        select_source_bin[m, :] = select_source & (bin_source[m] <= z_phot) & (z_phot < bin_source[m + 1])
    
    with h5py.File(os.path.join(comparison_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('select', data=select_source_bin, dtype=bool)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Comparison Select')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)