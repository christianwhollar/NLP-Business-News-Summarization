from datasets import Dataset
import os

class BuildFeatures():
    '''
    A class for building datasets from text and summary data files.

    Methods:
        __init__(self): Initializes the BuildFeatures class with file paths for train, validation, and test data.
        read_file(self, file_path): Reads the content of a file.

        build_path_lists(self, path): Builds lists of file paths for text and summary data in a given directory.
        build_dataset(self, text_data, summary_data): Builds a Hugging Face Dataset from text and summary data.
        get_datasets(self): Builds and returns datasets for train, validation, and test data.
    '''

    def __init__(self):
        '''
        Initializes the BuildFeatures class with file paths for train, validation, and test data.
        '''
        train_path = "data/processed/train/"
        val_path = "data/processed/val/"
        test_path = "data/processed/test/"
        self.path_dict = {'train': train_path, 'val': val_path, 'test': test_path}

    def read_file(self, file_path):
        '''
        Reads the content of a file.

        Args:
            file_path (str): The path of the file to read.

        Returns:
            str: The content of the file as a string.
        '''
        with open(file_path, 'r') as file:
            return file.read()
    
    def build_path_lists(self, path):
        '''
        Builds lists of file paths for text and summary data in a given directory.

        Args:
            path (str): The path of the directory containing text and summary data files.

        Returns:
            tuple: A tuple containing two lists - text_data (list of str) and summary_data (list of str).
        '''
        text_file_path = path + 'text/'
        summary_file_path = path + 'summaries/'

        text_paths = [os.path.join(text_file_path, file) for file in os.listdir(text_file_path)]
        summary_paths = [os.path.join(summary_file_path, file) for file in os.listdir(summary_file_path)]

        text_data = [self.read_file(file_path) for file_path in text_paths]
        summary_data = [self.read_file(file_path) for file_path in summary_paths]

        return text_data, summary_data
    
    def build_dataset(self, text_data, summary_data):
        '''
        Builds a Hugging Face Dataset from text and summary data.

        Args:
            text_data (list): A list of strings containing the text data.
            summary_data (list): A list of strings containing the summary data.

        Returns:
            datasets.Dataset: A Hugging Face Dataset containing the combined text and summary data.
        '''

        # Combine text and summary data into a dictionary
        data = {
            'text': text_data,
            'summary': summary_data
        }

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict(data)

        return dataset
    
    def get_datasets(self):
        '''
        Builds and returns datasets for train, validation, and test data.

        Returns:
            dict: A dictionary containing datasets for train, validation, and test data.
        '''
        dataset_dict = {}

        for name, path in self.path_dict.items():
            text_data, summary_data = self.build_path_lists(path)
            dataset_dict[name] = self.build_dataset(text_data, summary_data)

        return dataset_dict
