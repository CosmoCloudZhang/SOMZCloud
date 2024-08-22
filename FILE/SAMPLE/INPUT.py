import os
import time
import argparse


def main(path, folder, index):
    
    data_path = os.path.join(path, 'DATA/SAMPLE/')
    file_path = os.path.join(folder, 'cosmoDC2_gold_samples/training_samples/')
    
    data_name = os.path.join(data_path, 'INPUT_SAMPLE{}.hdf5'.format(index + 1))
    file_name = os.path.join(file_path, 'training_sample{}.hdf5'.format(index))
    
    os.system('cp -r {} {}'.format(file_name, data_name))
    
    return data_name, file_name


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Train samples.')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder.')
    PARSE.add_argument('--folder', type=str, required=True, help='The folder name of the datasets.')
    
    PATH = PARSE.parse_args().path
    FOLDER = PARSE.parse_args().folder
    
    LENGTH = 400
    for INDEX in range(LENGTH):
        print('Index: {}'.format(INDEX))
        
        START = time.time()
        RESULT = main(PATH, FOLDER, INDEX)
        
        END = time.time()
        print('Time: {:.2f} minutes'.format((END - START) / 60))
    
    DATA_PATH = os.path.join(PATH, 'DATA/SAMPLE/')
    FILE_PATH = os.path.join(FOLDER, 'cosmoDC2_gold_samples/')
    
    DATA_NAME = os.path.join(DATA_PATH, 'TEST_SAMPLE.hdf5')
    FILE_NAME = os.path.join(FILE_PATH, 'cosmoDC2_gold_test_catalog.hdf5')
    
    os.system('cp -r {} {}'.format(FILE_NAME, DATA_NAME))