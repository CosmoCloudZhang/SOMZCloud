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
    print('Index: {}'.format(index))
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    compare_folder = os.path.join(folder, 'COMPARE/')
    os.makedirs(os.path.join(compare_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(compare_folder, '{}/REFERENCE/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(compare_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(compare_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(compare_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(compare_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
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
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_sigma = file['morphology']['sigma'][...]
        degradation_magnitude = file['photometry']['mag_i_lsst'][...]
        degradation_redshift_true = file['photometry']['redshift_true'][...]
    degradation_size = len(degradation_magnitude)
    
    # Evaluate
    chunk_size = 100000
    evaluator = h5py.File(os.path.join(compare_folder, '{}/EVALUATE/EVALUATE{}.hdf5'.format(tag, index)), 'r')
    
    z_phot = numpy.zeros(degradation_size, dtype=numpy.float32)
    z_quantile = numpy.zeros(degradation_size, dtype=numpy.float32)
    
    for m in range(degradation_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, degradation_size)
        
        if begin < stop:
            z_pdf = numpy.maximum(scipy.interpolate.CubicSpline(x=z_grid, y=evaluator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate=False)(z_mesh), 0.0)
            z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
            z_phot[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
            
            z_cdf = numpy.cumsum(z_pdf, axis=1) * z_delta
            z_quantile[begin: stop] = z_cdf[numpy.arange(stop - begin), numpy.round((degradation_redshift_true[begin: stop] - z1) / z_delta).astype(numpy.int32)]
    
    # Reference Source
    sigma0 = 0.26
    reference_source = (z1_source <= z_phot) & (z_phot < z2_source) & (degradation_sigma < sigma0)
    
    # Reference Lens
    slope = 4.0
    intercept = 18.0
    reference_lens = (z1_lens <= z_phot) & (z_phot < z2_lens) & (degradation_magnitude < slope * z_phot + intercept)
    
    # Bin
    bin_size_lens = 10
    bin_lens = numpy.linspace(z1_lens, z2_lens, bin_size_lens + 1)
    
    bin_size_source = 5
    bin_source = numpy.quantile(z_phot[reference_source], numpy.linspace(0, 1, bin_size_source + 1))
    
    bin_source[0] = z1_source
    bin_source[-1] = z2_source
    
    # Point
    z_phot_lens = z_phot[reference_lens]
    z_true_lens = degradation_redshift_true[reference_lens]
    
    z_phot_source = z_phot[reference_source]
    z_true_source = degradation_redshift_true[reference_source]
    
    # Quantile
    z_quantile_lens = z_quantile[reference_lens]
    z_quantile_source = z_quantile[reference_source]
    
    # Histogram
    histogram_size = 10
    histogram_bin = numpy.linspace(0, 1, histogram_size + 1)
    
    # Metric
    delta = (z_phot - degradation_redshift_true) / (1 + degradation_redshift_true)
    
    bias = numpy.median(delta)
    sigma = scipy.stats.median_abs_deviation(delta)
    fraction = numpy.sum(numpy.abs(delta) > 0.15) / degradation_size
    rate = numpy.sum(numpy.abs(z_phot - degradation_redshift_true) > 1.0) / degradation_size
    
    histogram = numpy.histogram(z_quantile, bins=histogram_bin, range=(0, 1), density=True)[0]
    divergence = numpy.sqrt(numpy.sum(numpy.square(histogram - numpy.ones(histogram_size))) / histogram_size)
    
    # Metric Lens
    z1_average_lens = 0.0
    z2_average_lens = 1.5
    average_size_lens = 5
    z_average_lens = numpy.linspace(z1_average_lens, z2_average_lens, average_size_lens + 1)
    
    bias_lens = numpy.zeros(average_size_lens)
    rate_lens = numpy.zeros(average_size_lens)
    sigma_lens = numpy.zeros(average_size_lens)
    fraction_lens = numpy.zeros(average_size_lens)
    
    divergence_lens = numpy.zeros(average_size_lens)
    histogram_lens = numpy.zeros((average_size_lens, histogram_size))
    
    for m in range(average_size_lens):
        reference_lens_average = (z_average_lens[m] <= z_true_lens) & (z_true_lens < z_average_lens[m + 1])
        if numpy.sum(reference_lens_average) > 0:
            z_phot_lens_reference = z_phot_lens[reference_lens_average]
            z_true_lens_reference = z_true_lens[reference_lens_average]
            bias_lens_reference = (z_phot_lens_reference - z_true_lens_reference) / (1 + z_true_lens_reference)
            
            bias_lens[m] = numpy.median(bias_lens_reference)
            sigma_lens[m] = scipy.stats.median_abs_deviation(bias_lens_reference)
            fraction_lens[m] = numpy.sum(numpy.abs(bias_lens_reference) > 0.15) / numpy.sum(reference_lens_average)
            rate_lens[m] = numpy.sum(numpy.abs(z_phot_lens_reference - z_true_lens_reference) > 1.0) / numpy.sum(reference_lens_average)
            
            z_quantile_lens_reference = z_quantile_lens[reference_lens_average]
            histogram_lens[m, :] = numpy.histogram(z_quantile_lens_reference, bins=histogram_bin, range=(0, 1), density=True)[0]
            divergence_lens[m] = numpy.sqrt(numpy.sum(numpy.square(histogram_lens[m, :] - numpy.ones(histogram_size))) / histogram_size)
        else:
            bias_lens[m] = 0.0
            rate_lens[m] = 0.0
            sigma_lens[m] = 0.0
            fraction_lens[m] = 0.0
            
            divergence_lens[m] = 0.0
            histogram_lens[m, :] = 0.0
    
    # Metric Source
    z1_average_source = 0.0
    z2_average_source = 3.0
    average_size_source = 6
    z_average_source = numpy.linspace(z1_average_source, z2_average_source, average_size_source + 1)
    
    bias_source = numpy.zeros(average_size_source)
    rate_source = numpy.zeros(average_size_source)
    sigma_source = numpy.zeros(average_size_source)
    fraction_source = numpy.zeros(average_size_source)
    
    divergence_source = numpy.zeros(average_size_source)
    histogram_source = numpy.zeros((average_size_source, histogram_size))
    
    for m in range(average_size_source):
        reference_source_average = (z_average_source[m] <= z_true_source) & (z_true_source < z_average_source[m + 1])
        if numpy.sum(reference_source_average) > 0:
            z_phot_source_reference = z_phot_source[reference_source_average]
            z_true_source_reference = z_true_source[reference_source_average]
            bias_source_reference = (z_phot_source_reference - z_true_source_reference) / (1 + z_true_source_reference)
            
            bias_source[m] = numpy.median(bias_source_reference)
            sigma_source[m] = scipy.stats.median_abs_deviation(bias_source_reference)
            fraction_source[m] = numpy.sum(numpy.abs(bias_source_reference) > 0.15) / numpy.sum(reference_source_average)
            rate_source[m] = numpy.sum(numpy.abs(z_phot_source_reference - z_true_source_reference) > 1.0) / numpy.sum(reference_source_average)
            
            z_quantile_source_reference = z_quantile_source[reference_source_average]
            histogram_source[m, :] = numpy.histogram(z_quantile_source_reference, bins=histogram_bin, range=(0, 1), density=True)[0]
            divergence_source[m] = numpy.sqrt(numpy.sum(numpy.square(histogram_source[m, :] - numpy.ones(histogram_size))) / histogram_size)
        else:
            bias_source[m] = 0.0
            rate_source[m] = 0.0
            sigma_source[m] = 0.0
            fraction_source[m] = 0.0
            
            divergence_source[m] = 0.0
            histogram_source[m, :] = 0.0
    
    # Save
    with h5py.File(os.path.join(compare_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('bias', data=bias, dtype=numpy.float32)
        file.create_dataset('rate', data=rate, dtype=numpy.float32)
        file.create_dataset('sigma', data=sigma, dtype=numpy.float32)
        file.create_dataset('fraction', data=fraction, dtype=numpy.float32)
        
        file.create_dataset('histogram', data=histogram, dtype=numpy.float32)
        file.create_dataset('divergence', data=divergence, dtype=numpy.float32)
        
        file.create_dataset('z_phot', data=z_phot, dtype=numpy.float32)
        file.create_dataset('z_quantile', data=z_quantile, dtype=numpy.float32)
        
        # Lens
        file.create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file.create_dataset('reference_lens', data=reference_lens, dtype=bool)
        
        file.create_dataset('bias_lens', data=bias_lens, dtype=numpy.float32)
        file.create_dataset('rate_lens', data=rate_lens, dtype=numpy.float32)
        file.create_dataset('sigma_lens', data=sigma_lens, dtype=numpy.float32)
        file.create_dataset('fraction_lens', data=fraction_lens, dtype=numpy.float32)
        
        file.create_dataset('histogram_lens', data=histogram_lens, dtype=numpy.float32)
        file.create_dataset('divergence_lens', data=divergence_lens, dtype=numpy.float32)   
        
        # Source
        file.create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file.create_dataset('reference_source', data=reference_source, dtype=bool)
        
        file.create_dataset('bias_source', data=bias_source, dtype=numpy.float32)
        file.create_dataset('rate_source', data=rate_source, dtype=numpy.float32)
        file.create_dataset('sigma_source', data=sigma_source, dtype=numpy.float32)
        file.create_dataset('fraction_source', data=fraction_source, dtype=numpy.float32)
        
        file.create_dataset('histogram_source', data=histogram_source, dtype=numpy.float32)
        file.create_dataset('divergence_source', data=divergence_source, dtype=numpy.float32)
    
    # Lens bin
    reference_lens_bin = numpy.ones((bin_size_lens, degradation_size), dtype=bool)
    for m in range(len(bin_lens) - 1):
        reference_lens_bin[m, :] = reference_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
    
    with h5py.File(os.path.join(compare_folder, '{}/LENS/LENS{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('reference', data=reference_lens_bin, dtype=bool)
    
    # Source bin
    reference_source_bin = numpy.ones((bin_size_source, degradation_size), dtype=bool)
    for m in range(len(bin_source) - 1):
        reference_source_bin[m, :] = reference_source & (bin_source[m] <= z_phot) & (z_phot < bin_source[m + 1])
    
    with h5py.File(os.path.join(compare_folder, '{}/SOURCE/SOURCE{}/REFERENCE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('reference', data=reference_source_bin, dtype=bool)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Compare Reference')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)