from constants import *
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import shutil

class DownloadData():
    
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_API_KEY
    
    def download_kaggle(self, DATASET_NAMES = ['pariza/bbc-news-summary','rmisra/news-category-dataset'], PATH_NAME = './data/processed/'):
        api = KaggleApi()
        api.authenticate()
        for DATASET_NAME in DATASET_NAMES:
            print('Starting ' + DATASET_NAME + ' Download')
            api.dataset_download_files(DATASET_NAME, path = PATH_NAME, unzip=True)
            print('Completed ' + DATASET_NAME + ' Download')
    
    def move_data(self):
        letter_train_source = os.getcwd() + '/data/processed/asl_alphabet_train'
        number_train_source = os.getcwd() + '/data/processed/Train_Nums'
        train_destination = os.getcwd() + '/data/processed/train/'
        
        letter_test_source = os.getcwd() + '/data/processed/asl_alphabet_valid'
        number_test_source = os.getcwd() + '/data/processed/Test_Nums'
        test_destination = os.getcwd() + '/data/processed/test/'
        
        if not os.path.exists(train_destination):
            os.mkdir(train_destination)
            
        if not os.path.exists(test_destination):
            os.mkdir(test_destination)

        # gather all files
        allfiles = os.listdir(letter_train_source)
        
        for file in allfiles:
            if os.path.isdir(letter_train_source + '/' + file):
                shutil.move(letter_train_source + '/' + file,train_destination)
            
        # gather all files
        allfiles = os.listdir(number_train_source)
        
        for file in allfiles:
            if os.path.isdir(number_train_source + '/' + file):
                shutil.move(number_train_source + '/' + file,train_destination)
                
        # gather all files
        allfiles = os.listdir(letter_test_source)
        
        for file in allfiles:
            if os.path.isdir(letter_test_source + '/' + file):
                shutil.move(letter_test_source + '/' + file,test_destination)
            
        # gather all files
        allfiles = os.listdir(number_test_source)
        
        for file in allfiles:
            if os.path.isdir(number_test_source + '/' + file):
                shutil.move(number_test_source + '/' + file,test_destination)
        
        blank_dir = train_destination + 'Blank'
        if os.path.exists(blank_dir):
            shutil.rmtree(blank_dir)
            
        blank_dir = test_destination + 'Blank'
        if os.path.exists(blank_dir):
            shutil.rmtree(blank_dir)
            
        # # iterate on all files to move them to destination folder
        # for f in allfiles:
        #     src_path = os.path.join(source, f)
        #     dst_path = os.path.join(train_destination, f)
        #     os.rename(src_path, dst_path)
        # def download_obowflow(raw_dir):
        #     rf = Roboflow(api_key=ROBOFLOW_KEY, model_format="yolov7")
        #     rf.workspace().project(ROBOWFLOW_PROJECT).version(ROBOWFLOW_VERSION).download(location=raw_dir)