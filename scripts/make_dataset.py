from data.download_data import *
import os

class MakeDataset():
    '''
    MakeDataset Class for scripts/setup.py
    '''
    def __init__(self):
        '''
        Setup DownloadData class for Kaggle Dataset dowwnload
        Args:
            None
        '''
        self.dd = DownloadData()
        self.execute_download()
        self.gen_dataframes()
        self.gen_proc_dir()

    def execute_download(self):
        '''
        Call DownloadData to download Kaggle Datasets
        Args:
            None
        Returns:
            None
        '''
        print('Beginning download...')
        self.dd.download_kaggle()
        print('Download completed!')

    def gen_dataframes(self):   
        '''
        Call DownloadData to create pd.DataFrame for .json storage
        Args:
            None
        Returns:
            None
        '''
        if not hasattr(self.dd, 'dataframes'):
            self.dd.json_to_df()
        print('json converted to DataFrame!')
    
    def gen_proc_dir(self):
        '''
        Call DownloadData for URL scrape 
        Convert to .txt and move to /data/processed
        Args:
            None
        Returns:
            None
        '''
        if not os.listdir('data/processed/'):
            print('Downloading articles...')
            self.dd.download_to_text()
            print('Articles downloaded!')

            print('Moving and renaming files...')
            self.dd.move_rename_files()
            print('Shutil task completed!')