from data.download_data import *
import os

class MakeDataset():

    def __init__(self):
        '''
        
        '''
        self.dd = DownloadData()
        self.execute_download()
        self.gen_dataframes()
        self.gen_proc_dir()

    def execute_download(self):
        '''
        
        '''
        print('Beginning download...')
        self.dd.download_kaggle()
        print('Download completed!')

    def gen_dataframes(self):   
        '''
        
        '''
        if not hasattr(self.dd, 'dataframes'):
            self.dd.json_to_df()
        print('json converted to DataFrame!')
    
    def gen_proc_dir(self):
        '''
        
        '''
        if not os.listdir('data/processed/'):
            print('Downloading articles...')
            self.dd.download_to_text()
            print('Articles downloaded!')

            print('Moving and renaming files...')
            self.dd.move_rename_files()
            print('Shutil task completed!')