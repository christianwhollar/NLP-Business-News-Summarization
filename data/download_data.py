from bs4 import BeautifulSoup
from data.constants import *
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd
import requests
import shutil
from sklearn.model_selection import train_test_split

class DownloadData():
    
    def __init__(self):
        '''
        
        '''
        self.article_limit = 20
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_API_KEY
    
    def download_kaggle(self, DATASET_NAMES = ['pariza/bbc-news-summary','rmisra/news-category-dataset'], PATH_NAME = './data/raw/'):
        '''
        
        '''
        api = KaggleApi()
        api.authenticate()
        for DATASET_NAME in DATASET_NAMES:
            print('Starting ' + DATASET_NAME + ' Download')
            api.dataset_download_files(DATASET_NAME, path = PATH_NAME, unzip=True)
            print('Completed ' + DATASET_NAME + ' Download')
    
    def json_to_df(self, ext = 'News_Category_Dataset_v3.json'):
        '''
        
        '''
        raw_df = pd.read_json(f'data/raw/{ext}', lines=True)
        filtered_df = raw_df[raw_df['category'] == 'BUSINESS']
        filtered_df = filtered_df.head(self.article_limit)
        df = filtered_df.reset_index(drop=True)

        train_ratio = 0.7  # Percentage of data for training set
        val_ratio = 0.15  # Percentage of data for validation set
        test_ratio = 0.15  # Percentage of data for test set

        # Splitting into train and remaining data
        train_df, remaining_df = train_test_split(df, test_size=(1 - train_ratio), random_state=0)

        # Splitting remaining data into validation and test sets
        val_df, test_df = train_test_split(remaining_df, test_size=test_ratio/(test_ratio + val_ratio), random_state=0)

        # Optional: Reset the index for each DataFrame
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self.dataframes =  {'train': train_df, 'val': val_df, 'test': test_df}

        self.dataframe_sizes = {'train': train_ratio * self.article_limit, 
                                'val': val_ratio * self.article_limit,
                                'test': test_ratio * self.article_limit}
        
    def download_to_text(self):
        '''
        
        '''
        processed_dir = 'data/processed/'

        for df_name, df in self.dataframes.items():

            if not os.path.exists(processed_dir + df_name):
                # Create the directory
                os.makedirs(processed_dir + df_name)
                os.makedirs(processed_dir + df_name + '/text')
                os.makedirs(processed_dir + df_name + '/summaries')

            elif os.listdir(processed_dir + df_name + '/text') or os.listdir(processed_dir + df_name + '/summaries'):
                continue

            for idx, row in df.iterrows():

                filename = f"{idx:05d}.txt"

                url = row.link
                summary = row.short_description

                # Check if the substring is present in the string
                substring = "http"

                # Check if either substring is present in the string
                if substring in url:
                    # Find the indices of the last occurrences of the substrings
                    index = url.rfind(substring)
                    url = url[index:]

                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')

                # Extract body text from article
                bodytext = soup.find_all('p')
                bodytext = [i.text for i in bodytext]
                text = ' '.join(bodytext)

                with open(f'{processed_dir}{df_name}/text/' + filename, 'w', encoding='utf-8') as file:
                    file.write(text)

                with open(f'{processed_dir}{df_name}/summaries/' + filename, 'w', encoding='utf-8') as file:
                    file.write(summary)

    def get_largest_name_int(self, df_type = 'train'):

        example_dir = f"data/processed/{df_type}/text/"

        # Get the list of files in the directory
        file_list = os.listdir(example_dir)

        # Filter the list to include only text files
        txt_files = [file for file in file_list if file.endswith(".txt")]

        # Sort the list of text files in ascending order
        txt_files.sort()

        # Get the largest name
        largest_name = txt_files[-1]

        # Convert the largest name to an integer
        largest_name_int = int(largest_name.split(".")[0])

        return largest_name_int

    def move_rename_files(self):
        
        dir_tuple = [("data/raw/BBC News Summary/News Articles/business", "text"), 
                    ("data/raw/BBC News Summary/Summaries/business", "summaries")]

        # Move and rename the files
        for df_name in self.dataframes.keys():
            
            largest_name_int = self.get_largest_name_int(df_type = df_name)
            max_n = self.dataframe_sizes[df_name]

            for i in range(len(dir_tuple)):

                source_directory, target_ext = dir_tuple[i]

                target_directory = f"data/processed/{df_name}/{target_ext}"
                text_files = [file for file in os.listdir(source_directory) if file.endswith(".txt")]

                for index, file_name in enumerate(text_files, start=largest_name_int):

                    if index == max_n + largest_name_int:
                        break
                    
                    source_path = os.path.join(source_directory, file_name)
                    target_name = f"{index:05}.txt"
                    target_path = os.path.join(target_directory, target_name)
                    shutil.move(source_path, target_path)
                    # print(f"Moved and renamed {source_path} to {target_path}")