import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Define the lens and source reference
    
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
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(model_folder, '{}/REFERENCE/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(model_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(model_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(model_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
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
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_magnitude = file['photometry']['mag_i_lsst'][...]
        combination_redshift_true = file['photometry']['redshift_true'][...]
    combination_size = len(combination_magnitude)
    
    # Evaluate
    chunk_size = 100000
    z_phot = numpy.zeros(combination_size, dtype=numpy.float32)
    evaluator = h5py.File(os.path.join(model_folder, '{}/EVALUATE/EVALUATE{}.hdf5'.format(tag, index)), 'r')
    
    for m in range(combination_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, combination_size)
        
        if begin < stop:
            z_pdf = scipy.interpolate.CubicSpline(x=z_grid, y=evaluator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate=False)(z_mesh)
            z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
            z_phot[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
    
    # Reference
    slope = 4.0
    intercept = 18.0
    reference_source = (z1_source <= z_phot) & (z_phot < z2_source)
    reference_lens = (z1_lens <= z_phot) & (z_phot < z2_lens) & (combination_magnitude < slope * z_phot + intercept)
    
    # Bin
    lens_size = 5
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    
    bin_source = numpy.quantile(z_phot[reference_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Point
    z_phot_lens = z_phot[reference_lens]
    z_true_lens = combination_redshift_true[reference_lens]
    
    z_phot_source = z_phot[reference_source]
    z_true_source = combination_redshift_true[reference_source]
    
    # Metric Lens
    z1_average_lens = 0.0
    z2_average_lens = 1.6
    average_lens_size = 8
    z_average_lens = numpy.linspace(z1_average_lens, z2_average_lens, average_lens_size + 1)
    
    rate_lens = numpy.zeros(average_lens_size)
    delta_lens = numpy.zeros(average_lens_size)
    sigma_lens = numpy.zeros(average_lens_size)
    fraction_lens = numpy.zeros(average_lens_size)
    
    for m in range(average_lens_size):
        reference_lens_average = (z_average_lens[m] <= z_true_lens) & (z_true_lens < z_average_lens[m + 1])
        if numpy.sum(reference_lens_average) > 0:
            z_phot_lens_select = z_phot_lens[reference_lens_average]
            z_true_lens_select = z_true_lens[reference_lens_average]
            delta_lens_select = numpy.abs(z_phot_lens_select - z_true_lens_select) / (1 + z_true_lens_select)
            
            delta_lens[m] = numpy.median(delta_lens_select)
            fraction_lens[m] = len(delta_lens_select[delta_lens_select > 0.15]) / len(delta_lens_select)
            sigma_lens[m] = 1.4826 * numpy.median(numpy.abs(delta_lens_select - numpy.median(delta_lens_select)))
            rate_lens[m] = len(delta_lens_select[numpy.abs(z_phot_lens_select - z_true_lens_select) > 1.0]) / len(delta_lens_select)
        else:
            rate_lens[m] = numpy.nan
            delta_lens[m] = numpy.nan
            sigma_lens[m] = numpy.nan
            fraction_lens[m] = numpy.nan
    
    # Metric Source
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_source_size = 10
    z_average_source = numpy.linspace(z1_average_source, z2_average_source, average_source_size + 1)
    
    rate_source = numpy.zeros(average_source_size)
    delta_source = numpy.zeros(average_source_size)
    sigma_source = numpy.zeros(average_source_size)
    fraction_source = numpy.zeros(average_source_size)
    
    for m in range(average_source_size):
        reference_source_average = (z_average_source[m] <= z_true_source) & (z_true_source < z_average_source[m + 1])
        if numpy.sum(reference_source_average) > 0:
            z_phot_source_select = z_phot_source[reference_source_average]
            z_true_source_select = z_true_source[reference_source_average]
            delta_source_select = numpy.abs(z_phot_source_select - z_true_source_select) / (1 + z_true_source_select)
            
            delta_source[m] = numpy.median(delta_source_select)
            fraction_source[m] = len(delta_source_select[delta_source_select > 0.15]) / len(delta_source_select)
            sigma_source[m] = 1.4826 * numpy.median(numpy.abs(delta_source_select - numpy.median(delta_source_select)))
            rate_source[m] = len(delta_source_select[numpy.abs(z_phot_source_select - z_true_source_select) > 1.0]) / len(delta_source_select)
        else:
            rate_source[m] = numpy.nan
            delta_source[m] = numpy.nan
            sigma_source[m] = numpy.nan
            fraction_source[m] = numpy.nan
    
    # Save
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('z_phot', data=z_phot, dtype=numpy.float32)
        
        file.create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file.create_dataset('reference_lens', data=reference_lens, dtype=bool)
        
        file.create_dataset('rate_lens', data=rate_lens, dtype=numpy.float32)
        file.create_dataset('delta_lens', data=delta_lens, dtype=numpy.float32)
        file.create_dataset('sigma_lens', data=sigma_lens, dtype=numpy.float32)
        file.create_dataset('fraction_lens', data=fraction_lens, dtype=numpy.float32)
        
        file.create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file.create_dataset('reference_source', data=reference_source, dtype=bool)
        
        file.create_dataset('rate_source', data=rate_source, dtype=numpy.float32)
        file.create_dataset('delta_source', data=delta_source, dtype=numpy.float32)
        file.create_dataset('sigma_source', data=sigma_source, dtype=numpy.float32)
        file.create_dataset('fraction_source', data=fraction_source, dtype=numpy.float32)
    
    # Lens bin
    reference_lens_bin = numpy.ones((lens_size, combination_size), dtype=bool)
    for m in range(len(bin_lens) - 1):
        reference_lens_bin[m, :] = reference_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
    
    with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('reference', data=reference_lens_bin, dtype=bool)
    
    # Source
    reference_source_bin = numpy.ones((source_size, combination_size), dtype=bool)
    for m in range(len(bin_source) - 1):
        reference_source_bin[m, :] = reference_source & (bin_source[m] <= z_phot) & (z_phot < bin_source[m + 1])
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('reference', data=reference_source_bin, dtype=bool)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Model Reference')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)