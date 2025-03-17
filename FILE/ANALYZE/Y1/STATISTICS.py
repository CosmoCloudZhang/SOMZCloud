import os
import h5py
import time
import numpy
import scipy
import argparse


def main(tag, folder):
    '''
    This function is used to analyze the information of the dataset
    
    Arguments:
        tag (str): The tag of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/STATISTICS/'.format(tag)), exist_ok=True)
    
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
    for label in label_list:
        
        # Summarize
        with h5py.File(os.path.join(synthesize_folder, '{}/SOM_{}.hdf5'.format(tag, label)), 'r') as file:
            som_data_lens = file['lens']['data'][...]
            som_data_source = file['source']['data'][...]
            
            som_average_lens = file['lens']['average'][...]
            som_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/MODEL_{}.hdf5'.format(tag, label)), 'r') as file:
            model_data_lens = file['lens']['data'][...]
            model_data_source = file['source']['data'][...]
            
            model_average_lens = file['lens']['average'][...]
            model_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
            product_data_lens = file['lens']['data'][...]
            product_data_source = file['source']['data'][...]
            
            product_average_lens = file['lens']['average'][...]
            product_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/FIDUCIAL_{}.hdf5'.format(tag, label)), 'r') as file:
            fiducial_data_lens = file['lens']['data'][...]
            fiducial_data_source = file['source']['data'][...]
            
            fiducial_average_lens = file['lens']['average'][...]
            fiducial_average_source = file['source']['average'][...]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/HISTOGRAM_{}.hdf5'.format(tag, label)), 'r') as file:
            histogram_data_lens = file['lens']['data'][...]
            histogram_data_source = file['source']['data'][...]
            
            histogram_average_lens = file['lens']['average'][...]
            histogram_average_source = file['source']['average'][...]
        
        data_size, bin_lens_size, z_size = histogram_data_lens.shape
        data_size, bin_source_size, z_size = histogram_data_source.shape
        
        # Redshift
        z1 = 0.0
        z2 = 3.0
        z_grid = numpy.linspace(z1, z2, z_size)
        
        # Expectation
        som_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_data_lens, axis=2)
        som_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * som_data_source, axis=2)
        
        model_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_data_lens, axis=2)
        model_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * model_data_source, axis=2)
        
        product_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_data_lens, axis=2)
        product_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * product_data_source, axis=2)
        
        fiducial_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_data_lens, axis=2)
        fiducial_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * fiducial_data_source, axis=2)
        
        histogram_expectation_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_data_lens, axis=2)
        histogram_expectation_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, numpy.newaxis, :] * histogram_data_source, axis=2)
        
        # Middle
        som_middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_average_lens, axis=1)
        som_middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * som_average_source, axis=1)
        
        model_middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_average_lens, axis=1)
        model_middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * model_average_source, axis=1)
        
        product_middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_average_lens, axis=1)
        product_middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * product_average_source, axis=1)
        
        fiducial_middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * fiducial_average_lens, axis=1)
        fiducial_middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * fiducial_average_source, axis=1)
        
        histogram_middle_lens = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_average_lens, axis=1)
        histogram_middle_source = scipy.integrate.trapezoid(x=z_grid, y=z_grid[numpy.newaxis, :] * histogram_average_source, axis=1)
        
        # Deviation
        som_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - som_expectation_lens[:, :, numpy.newaxis]) * som_data_lens, axis=2))
        som_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - som_expectation_source[:, :, numpy.newaxis]) * som_data_source, axis=2))
        
        model_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - model_expectation_lens[:, :, numpy.newaxis]) * model_data_lens, axis=2))
        model_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - model_expectation_source[:, :, numpy.newaxis]) * model_data_source, axis=2))
        
        product_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - product_expectation_lens[:, :, numpy.newaxis]) * product_data_lens, axis=2))
        product_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - product_expectation_source[:, :, numpy.newaxis]) * product_data_source, axis=2))
        
        fiducial_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - fiducial_expectation_lens[:, :, numpy.newaxis]) * fiducial_data_lens, axis=2))
        fiducial_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - fiducial_expectation_source[:, :, numpy.newaxis]) * fiducial_data_source, axis=2))
        
        histogram_deviation_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - histogram_expectation_lens[:, :, numpy.newaxis]) * histogram_data_lens, axis=2))
        histogram_deviation_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, numpy.newaxis, :] - histogram_expectation_source[:, :, numpy.newaxis]) * histogram_data_source, axis=2))
        
        # Scatter
        som_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - som_middle_lens[:, numpy.newaxis]) * som_average_lens, axis=1))
        som_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - som_middle_source[:, numpy.newaxis]) * som_average_source, axis=1))
        
        model_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - model_middle_lens[:, numpy.newaxis]) * model_average_lens, axis=1))
        model_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - model_middle_source[:, numpy.newaxis]) * model_average_source, axis=1))
        
        product_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - product_middle_lens[:, numpy.newaxis]) * product_average_lens, axis=1))
        product_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - product_middle_source[:, numpy.newaxis]) * product_average_source, axis=1))
        
        fiducial_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - fiducial_middle_lens[:, numpy.newaxis]) * fiducial_average_lens, axis=1))
        fiducial_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - fiducial_middle_source[:, numpy.newaxis]) * fiducial_average_source, axis=1))
        
        histogram_scatter_lens = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - histogram_middle_lens[:, numpy.newaxis]) * histogram_average_lens, axis=1))
        histogram_scatter_source = numpy.sqrt(scipy.integrate.trapezoid(x=z_grid, y=numpy.square(z_grid[numpy.newaxis, :] - histogram_middle_source[:, numpy.newaxis]) * histogram_average_source, axis=1))
        
        # Scale
        som_scale_lens = numpy.std(som_expectation_lens, axis=0) / (1 + som_middle_lens)
        som_scale_source = numpy.std(som_expectation_source, axis=0) / (1 + som_middle_source)
        
        model_scale_lens = numpy.std(model_expectation_lens, axis=0) / (1 + model_middle_lens)
        model_scale_source = numpy.std(model_expectation_source, axis=0) / (1 + model_middle_source)
        
        product_scale_lens = numpy.std(product_expectation_lens, axis=0) / (1 + product_middle_lens)
        product_scale_source = numpy.std(product_expectation_source, axis=0) / (1 + product_middle_source)
        
        fiducial_scale_lens = numpy.std(fiducial_expectation_lens, axis=0) / (1 + fiducial_middle_lens)
        fiducial_scale_source = numpy.std(fiducial_expectation_source, axis=0) / (1 + fiducial_middle_source)
        
        histogram_scale_lens = numpy.std(histogram_expectation_lens, axis=0) / (1 + histogram_middle_lens)
        histogram_scale_source = numpy.std(histogram_expectation_source, axis=0) / (1 + histogram_middle_source)
        
        # Variation
        som_variation_lens = numpy.std(som_deviation_lens, axis=0) / (1 + som_middle_lens)
        som_variation_source = numpy.std(som_deviation_source, axis=0) / (1 + som_middle_source)
        
        model_variation_lens = numpy.std(model_deviation_lens, axis=0) / (1 + model_middle_lens)
        model_variation_source = numpy.std(model_deviation_source, axis=0) / (1 + model_middle_source)
        
        product_variation_lens = numpy.std(product_deviation_lens, axis=0) / (1 + product_middle_lens)
        product_variation_source = numpy.std(product_deviation_source, axis=0) / (1 + product_middle_source)
        
        fiducial_variation_lens = numpy.std(fiducial_deviation_lens, axis=0) / (1 + fiducial_middle_lens)
        fiducial_variation_source = numpy.std(fiducial_deviation_source, axis=0) / (1 + fiducial_middle_source)
        
        histogram_variation_lens = numpy.std(histogram_deviation_lens, axis=0) / (1 + histogram_middle_lens)
        histogram_variation_source = numpy.std(histogram_deviation_source, axis=0) / (1 + histogram_middle_source)
        
        # Correlation
        som_correlation_lens = numpy.corrcoef(som_expectation_lens, rowvar=False)
        som_correlation_source = numpy.corrcoef(som_expectation_source, rowvar=False)
        
        model_correlation_lens = numpy.corrcoef(model_expectation_lens, rowvar=False)
        model_correlation_source = numpy.corrcoef(model_expectation_source, rowvar=False)
        
        product_correlation_lens = numpy.corrcoef(product_expectation_lens, rowvar=False)
        product_correlation_source = numpy.corrcoef(product_expectation_source, rowvar=False)
        
        fiducial_correlation_lens = numpy.corrcoef(fiducial_expectation_lens, rowvar=False)
        fiducial_correlation_source = numpy.corrcoef(fiducial_expectation_source, rowvar=False)
        
        histogram_correlation_lens = numpy.corrcoef(histogram_expectation_lens, rowvar=False)
        histogram_correlation_source = numpy.corrcoef(histogram_expectation_source, rowvar=False)
        
        # Delta
        som_delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(som_expectation_lens, rowvar=False), size=data_size)
        som_delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(som_expectation_source, rowvar=False), size=data_size)
        
        model_delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(model_expectation_lens, rowvar=False), size=data_size)
        model_delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(model_expectation_source, rowvar=False), size=data_size)
        
        product_delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(product_expectation_lens, rowvar=False), size=data_size)
        product_delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(product_expectation_source, rowvar=False), size=data_size)
        
        fiducial_delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(fiducial_expectation_lens, rowvar=False), size=data_size)
        fiducial_delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(fiducial_expectation_source, rowvar=False), size=data_size)
        
        histogram_delta_lens = numpy.random.multivariate_normal(mean=numpy.zeros(bin_lens_size), cov=numpy.cov(histogram_expectation_lens, rowvar=False), size=data_size)
        histogram_delta_source = numpy.random.multivariate_normal(mean=numpy.zeros(bin_source_size), cov=numpy.cov(histogram_expectation_source, rowvar=False), size=data_size)
        
        # Shift
        som_shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
        som_shift_source = numpy.zeros((data_size, bin_source_size, z_size))
        
        model_shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
        model_shift_source = numpy.zeros((data_size, bin_source_size, z_size))
        
        product_shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
        product_shift_source = numpy.zeros((data_size, bin_source_size, z_size))
        
        fiducial_shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
        fiducial_shift_source = numpy.zeros((data_size, bin_source_size, z_size))
        
        histogram_shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
        histogram_shift_source = numpy.zeros((data_size, bin_source_size, z_size))
        
        for m in range(bin_lens_size):
            som_shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, som_average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - som_delta_lens[:, m, numpy.newaxis]), 0)
            
            model_shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, model_average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - model_delta_lens[:, m, numpy.newaxis]), 0)
            
            product_shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, product_average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - product_delta_lens[:, m, numpy.newaxis]), 0)
            
            fiducial_shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, fiducial_average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - fiducial_delta_lens[:, m, numpy.newaxis]), 0)
            
            histogram_shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, histogram_average_lens[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - histogram_delta_lens[:, m, numpy.newaxis]), 0)
        
        som_shift_lens = som_shift_lens / scipy.integrate.trapezoid(x=z_grid, y=som_shift_lens, axis=2)[:, :, numpy.newaxis]
        
        model_shift_lens = model_shift_lens / scipy.integrate.trapezoid(x=z_grid, y=model_shift_lens, axis=2)[:, :, numpy.newaxis]
        
        product_shift_lens = product_shift_lens / scipy.integrate.trapezoid(x=z_grid, y=product_shift_lens, axis=2)[:, :, numpy.newaxis]
        
        fiducial_shift_lens = fiducial_shift_lens / scipy.integrate.trapezoid(x=z_grid, y=fiducial_shift_lens, axis=2)[:, :, numpy.newaxis]
        
        histogram_shift_lens = histogram_shift_lens / scipy.integrate.trapezoid(x=z_grid, y=histogram_shift_lens, axis=2)[:, :, numpy.newaxis]
        
        for m in range(bin_source_size):
            som_shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, som_average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - som_delta_source[:, m, numpy.newaxis]), 0)
            
            model_shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, model_average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - model_delta_source[:, m, numpy.newaxis]), 0)
            
            product_shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, product_average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - product_delta_source[:, m, numpy.newaxis]), 0)
            
            fiducial_shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, fiducial_average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - fiducial_delta_source[:, m, numpy.newaxis]), 0)
            
            histogram_shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, histogram_average_source[m, :], extrapolate=True)(z_grid[numpy.newaxis, :] - histogram_delta_source[:, m, numpy.newaxis]), 0)
        
        som_shift_source = som_shift_source / scipy.integrate.trapezoid(x=z_grid, y=som_shift_source, axis=2)[:, :, numpy.newaxis]
        
        model_shift_source = model_shift_source / scipy.integrate.trapezoid(x=z_grid, y=model_shift_source, axis=2)[:, :, numpy.newaxis]
        
        product_shift_source = product_shift_source / scipy.integrate.trapezoid(x=z_grid, y=product_shift_source, axis=2)[:, :, numpy.newaxis]
        
        fiducial_shift_source = fiducial_shift_source / scipy.integrate.trapezoid(x=z_grid, y=fiducial_shift_source, axis=2)[:, :, numpy.newaxis]
        
        histogram_shift_source = histogram_shift_source / scipy.integrate.trapezoid(x=z_grid, y=histogram_shift_source, axis=2)[:, :, numpy.newaxis]
        
        # Save
        with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/SOM_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('delta', data=som_delta_lens)
            file['lens'].create_dataset('scale', data=som_scale_lens)
            file['lens'].create_dataset('shift', data=som_shift_lens)
            file['lens'].create_dataset('middle', data=som_middle_lens)
            file['lens'].create_dataset('scatter', data=som_scatter_lens)
            file['lens'].create_dataset('deviation', data=som_deviation_lens)
            file['lens'].create_dataset('variation', data=som_variation_lens)
            file['lens'].create_dataset('expectation', data=som_expectation_lens)
            file['lens'].create_dataset('correlation', data=som_correlation_lens)
            
            file.create_group('source')
            file['source'].create_dataset('delta', data=som_delta_source)
            file['source'].create_dataset('scale', data=som_scale_source)
            file['source'].create_dataset('shift', data=som_shift_source)
            file['source'].create_dataset('middle', data=som_middle_source)
            file['source'].create_dataset('scatter', data=som_scatter_source)
            file['source'].create_dataset('deviation', data=som_deviation_source)
            file['source'].create_dataset('variation', data=som_variation_source)
            file['source'].create_dataset('expectation', data=som_expectation_source)
            file['source'].create_dataset('correlation', data=som_correlation_source)
        
        with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/MODEL_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('delta', data=model_delta_lens)
            file['lens'].create_dataset('scale', data=model_scale_lens)
            file['lens'].create_dataset('shift', data=model_shift_lens)
            file['lens'].create_dataset('middle', data=model_middle_lens)
            file['lens'].create_dataset('scatter', data=model_scatter_lens)
            file['lens'].create_dataset('deviation', data=model_deviation_lens)
            file['lens'].create_dataset('variation', data=model_variation_lens)
            file['lens'].create_dataset('expectation', data=model_expectation_lens)
            file['lens'].create_dataset('correlation', data=model_correlation_lens)
            
            file.create_group('source')
            file['source'].create_dataset('delta', data=model_delta_source)
            file['source'].create_dataset('scale', data=model_scale_source)
            file['source'].create_dataset('shift', data=model_shift_source)
            file['source'].create_dataset('middle', data=model_middle_source)
            file['source'].create_dataset('scatter', data=model_scatter_source)
            file['source'].create_dataset('deviation', data=model_deviation_source)
            file['source'].create_dataset('variation', data=model_variation_source)
            file['source'].create_dataset('expectation', data=model_expectation_source)
            file['source'].create_dataset('correlation', data=model_correlation_source)
        
        with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/PRODUCT_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('delta', data=product_delta_lens)
            file['lens'].create_dataset('scale', data=product_scale_lens)
            file['lens'].create_dataset('shift', data=product_shift_lens)
            file['lens'].create_dataset('middle', data=product_middle_lens)
            file['lens'].create_dataset('scatter', data=product_scatter_lens)
            file['lens'].create_dataset('deviation', data=product_deviation_lens)
            file['lens'].create_dataset('variation', data=product_variation_lens)
            file['lens'].create_dataset('expectation', data=product_expectation_lens)
            file['lens'].create_dataset('correlation', data=product_correlation_lens)
            
            file.create_group('source')
            file['source'].create_dataset('delta', data=product_delta_source)
            file['source'].create_dataset('scale', data=product_scale_source)
            file['source'].create_dataset('shift', data=product_shift_source)
            file['source'].create_dataset('middle', data=product_middle_source)
            file['source'].create_dataset('scatter', data=product_scatter_source)
            file['source'].create_dataset('deviation', data=product_deviation_source)
            file['source'].create_dataset('variation', data=product_variation_source)
            file['source'].create_dataset('expectation', data=product_expectation_source)
            file['source'].create_dataset('correlation', data=product_correlation_source)
        
        with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/FIDUCIAL_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('delta', data=fiducial_delta_lens)
            file['lens'].create_dataset('scale', data=fiducial_scale_lens)
            file['lens'].create_dataset('shift', data=fiducial_shift_lens)
            file['lens'].create_dataset('middle', data=fiducial_middle_lens)
            file['lens'].create_dataset('scatter', data=fiducial_scatter_lens)
            file['lens'].create_dataset('deviation', data=fiducial_deviation_lens)
            file['lens'].create_dataset('variation', data=fiducial_variation_lens)
            file['lens'].create_dataset('expectation', data=fiducial_expectation_lens)
            file['lens'].create_dataset('correlation', data=fiducial_correlation_lens)
            
            file.create_group('source')
            file['source'].create_dataset('delta', data=fiducial_delta_source)
            file['source'].create_dataset('scale', data=fiducial_scale_source)
            file['source'].create_dataset('shift', data=fiducial_shift_source)
            file['source'].create_dataset('middle', data=fiducial_middle_source)
            file['source'].create_dataset('scatter', data=fiducial_scatter_source)
            file['source'].create_dataset('deviation', data=fiducial_deviation_source)
            file['source'].create_dataset('variation', data=fiducial_variation_source)
            file['source'].create_dataset('expectation', data=fiducial_expectation_source)
            file['source'].create_dataset('correlation', data=fiducial_correlation_source)
        
        with h5py.File(os.path.join(analyze_folder, '{}/STATISTICS/HISTOGRAM_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('delta', data=histogram_delta_lens)
            file['lens'].create_dataset('scale', data=histogram_scale_lens)
            file['lens'].create_dataset('shift', data=histogram_shift_lens)
            file['lens'].create_dataset('middle', data=histogram_middle_lens)
            file['lens'].create_dataset('scatter', data=histogram_scatter_lens)
            file['lens'].create_dataset('deviation', data=histogram_deviation_lens)
            file['lens'].create_dataset('variation', data=histogram_variation_lens)
            file['lens'].create_dataset('expectation', data=histogram_expectation_lens)
            file['lens'].create_dataset('correlation', data=histogram_correlation_lens)
            
            file.create_group('source')
            file['source'].create_dataset('delta', data=histogram_delta_source)
            file['source'].create_dataset('scale', data=histogram_scale_source)
            file['source'].create_dataset('shift', data=histogram_shift_source)
            file['source'].create_dataset('middle', data=histogram_middle_source)
            file['source'].create_dataset('scatter', data=histogram_scatter_source)
            file['source'].create_dataset('deviation', data=histogram_deviation_source)
            file['source'].create_dataset('variation', data=histogram_variation_source)
            file['source'].create_dataset('expectation', data=histogram_expectation_source)
            file['source'].create_dataset('correlation', data=histogram_correlation_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analysis Statistics')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, FOLDER)