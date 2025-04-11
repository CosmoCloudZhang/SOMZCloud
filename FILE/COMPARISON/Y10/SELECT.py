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
    z_phot = numpy.zeros(application_size, dtype=numpy.float32)
    estimator = h5py.File(os.path.join(comparison_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    for m in range(application_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, application_size)
        
        if begin < stop:
            z_pdf = scipy.interpolate.CubicSpline(x=z_grid, y=estimator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate=False)(z_mesh)
            z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
            z_phot[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = (z1_source <= z_phot) & (z_phot < z2_source)
    select_lens = (z1_lens <= z_phot) & (z_phot < z2_lens) & (application_magnitude < slope * z_phot + intercept)
    
    # Bin
    lens_size = 10
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    
    bin_source = numpy.quantile(z_phot[select_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Point
    z_phot_lens = z_phot[select_lens]
    z_true_lens = application_redshift_true[select_lens]
    
    z_phot_source = z_phot[select_source]
    z_true_source = application_redshift_true[select_source]
    
    # Metric Lens
    average_lens_size = 6
    z_average_lens = numpy.linspace(z1_lens, z2_lens, average_lens_size + 1)
    
    nmad_lens = numpy.zeros(average_lens_size)
    fraction_lens = numpy.zeros(average_lens_size)
    percentile_lens = numpy.zeros(average_lens_size)
    delta_average_lens = numpy.zeros(average_lens_size)
    
    for m in range(average_lens_size):
        select_lens_average = (z_average_lens[m] <= z_true_lens) & (z_true_lens < z_average_lens[m + 1])
        if numpy.sum(select_lens_average) > 0:
            z_phot_lens_select = z_phot_lens[select_lens_average]
            z_true_lens_select = z_true_lens[select_lens_average]
            delta_lens_select = numpy.abs(z_phot_lens_select - z_true_lens_select) / (1 + z_true_lens_select)
            
            delta_average_lens[m] = numpy.mean(delta_lens_select)
            fraction_lens[m] = len(delta_lens_select[delta_lens_select > 0.15]) / len(delta_lens_select)
            nmad_lens[m] = 1.4826 * numpy.median(numpy.abs(delta_lens_select - numpy.median(delta_lens_select)))
            percentile_lens[m] = len(delta_lens_select[numpy.abs(z_phot_lens_select - z_true_lens_select) > 1.0]) / len(delta_lens_select) * 100
        else:
            nmad_lens[m] = numpy.nan
            fraction_lens[m] = numpy.nan
            percentile_lens[m] = numpy.nan
            delta_average_lens[m] = numpy.nan
    
    # Metric Source
    average_source_size = 6
    z_average_source = numpy.linspace(z1, z2, average_source_size + 1)
    
    nmad_source = numpy.zeros(average_source_size)
    fraction_source = numpy.zeros(average_source_size)
    percentile_source = numpy.zeros(average_source_size)
    delta_average_source = numpy.zeros(average_source_size)
    
    for m in range(average_source_size):
        select_source_average = (z_average_source[m] <= z_true_source) & (z_true_source < z_average_source[m + 1])
        if numpy.sum(select_source_average) > 0:
            z_phot_source_select = z_phot_source[select_source_average]
            z_true_source_select = z_true_source[select_source_average]
            delta_source_select = numpy.abs(z_phot_source_select - z_true_source_select) / (1 + z_true_source_select)
            
            delta_average_source[m] = numpy.mean(delta_source_select)
            fraction_source[m] = len(delta_source_select[delta_source_select > 0.15]) / len(delta_source_select)
            nmad_source[m] = 1.4826 * numpy.median(numpy.abs(delta_source_select - numpy.median(delta_source_select)))
            percentile_source[m] = len(delta_source_select[numpy.abs(z_phot_source_select - z_true_source_select) > 1.0]) / len(delta_source_select) * 100
        else:
            nmad_source[m] = numpy.nan
            fraction_source[m] = numpy.nan
            percentile_source[m] = numpy.nan
            delta_average_source[m] = numpy.nan
    
    # Save
    with h5py.File(os.path.join(comparison_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('z_phot', data=z_phot, dtype=numpy.float32)
        
        file.create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file.create_dataset('select_lens', data=select_lens, dtype=bool)
        
        file.create_dataset('nmad_lens', data=nmad_lens, dtype=numpy.float32)
        file.create_dataset('fraction_lens', data=fraction_lens, dtype=numpy.float32)
        file.create_dataset('percentile_lens', data=percentile_lens, dtype=numpy.float32)
        file.create_dataset('delta_average_lens', data=delta_average_lens, dtype=numpy.float32)
        
        file.create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file.create_dataset('select_source', data=select_source, dtype=bool)
        
        file.create_dataset('nmad_source', data=nmad_source, dtype=numpy.float32)
        file.create_dataset('fraction_source', data=fraction_source, dtype=numpy.float32)
        file.create_dataset('percentile_source', data=percentile_source, dtype=numpy.float32)
        file.create_dataset('delta_average_source', data=delta_average_source, dtype=numpy.float32)
    
    # Lens bin
    select_lens_bin = numpy.ones((lens_size, application_size), dtype=bool)
    for m in range(len(bin_lens) - 1):
        select_lens_bin[m, :] = select_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
    
    with h5py.File(os.path.join(comparison_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('select', data=select_lens_bin, dtype=bool)
    
    # Source
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
    PARSE = argparse.ArgumentParser(description='Model Select')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)