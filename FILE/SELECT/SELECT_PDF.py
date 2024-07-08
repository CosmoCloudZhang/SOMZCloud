import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing

def save_pdf(z_lens, z_pdf, z_mean, z_source, bin_lens, bin_source, mag0_lens, mag_source):
    """
    Save the selected samples.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_pdf (numpy.ndarray): The redshift PDF of source samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        bin_lens (numpy.ndarray): The redshift bin of lens samples.
        bin_source (numpy.ndarray): The redshift bin of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples.
    
    Returns:
        tuple: The selected lens and source samples.
    """
    # Select
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    slope = 4.0
    intersection = 18.0
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intersection) & (mag_source < mag0_lens)
    meta = {'pdf_name': numpy.array(['interp'.encode('ascii')]).astype('S6'), 'pdf_version': numpy.array([0]).astype(numpy.int32), 'xvals': numpy.array([z_source]).astype(numpy.float32)}
    
    # Lens
    lens = []
    lens_size = len(bin_lens) - 1
    for m in range(lens_size):
        select = select_lens & (bin_lens[m] < z_mean) & (z_mean < bin_lens[m + 1])
        data = {'yvals': z_pdf[select, :].astype(numpy.float32)}
        lens.append({'data': data, 'meta': meta})
    
    # Source
    source = []
    source_size = len(bin_source) - 1
    for m in range(source_size):
        select = select_source & (bin_source[m] < z_mean) & (z_mean < bin_source[m + 1])
        data = {'yvals': z_pdf[select, :].astype(numpy.float32)}
        source.append({'data': data, 'meta': meta})
    
    # Return
    return lens, source


def main(path, index):
    start = time.time()
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'ESTIMATE/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Bin
    bin_name = os.path.join(data_path, 'BIN/BIN.hdf5')
    with h5py.File(bin_name, 'r') as file:
        bin_lens = file['lens'][:].astype(numpy.float32)
        bin_source = file['source'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.0
    z2_lens = 2.0
    z_lens_size = 200
    z_lens = numpy.linspace(z1_lens, z2_lens, z_lens_size + 1)
    
    z1_source = 0.0
    z2_source = 3.0
    z_source_size = 300
    z_source = numpy.linspace(z1_source, z2_source, z_source_size + 1)
    
    z_pdf = estimator().pdf(z_source)
    z_mean = numpy.concatenate(estimator().mean())
    mag_source = test_data()['photometry']['mag_i_lsst']
    
    # Magnitude
    mag0_lens = 24.1
    
    # Save PDF
    lens_pdf, source_pdf = save_pdf(z_lens, z_pdf, z_mean, z_source, bin_lens, bin_source, mag0_lens, mag_source)
    
    # Lens
    lens_size = len(bin_lens) - 1
    os.makedirs(os.path.join(data_path, 'LENS/LENS{}'.format(index)), exist_ok=True)
    
    for m in range(lens_size):        
        with h5py.File(os.path.join(data_path, 'LENS/LENS{}/SELECT_PDF{}.hdf5'.format(index, m)), 'w') as file:
            for name in lens_pdf[m].keys():
                file.create_group(name)
                for key, value in lens_pdf[m][name].items():
                    file[name].create_dataset(key, data=value)
    
    # Source
    source_size = len(bin_source) - 1
    os.makedirs(os.path.join(data_path, 'SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    
    for n in range(source_size):
        with h5py.File(os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT_PDF{}.hdf5'.format(index, n)), 'w') as file:
            for name in source_pdf[n].keys():
                file.create_group(name)
                for key, value in source_pdf[n][name].items():
                    file[name].create_dataset(key, data=value)
    
    # Return
    end = time.time()
    print('Index:{}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return index

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])