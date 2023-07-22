from datasets import Dataset
import os

class BuildFeatures():
    '''

    '''
    def __init__(self):
        train_path = "data/processed/train/"
        val_path = "data/processed/val/"
        test_path = "data/processed/test/"
        self.path_dict = {'train': train_path, 'val': val_path, 'test': test_path}

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()
    
    def build_path_lists(self, path):
        '''
        
        '''
        text_file_path = path + 'text/'
        summary_file_path = path + 'summaries/'

        text_paths = [os.path.join(text_file_path, file) for file in os.listdir(text_file_path)]
        summary_paths = [os.path.join(summary_file_path, file) for file in os.listdir(summary_file_path)]

        text_data = [self.read_file(file_path) for file_path in text_paths]
        summary_data = [self.read_file(file_path) for file_path in summary_paths]

        return text_data, summary_data
    
    def build_dataset(self, text_data, summary_data):
        # Combine text and summary data into a dictionary
        data = {
            'text': text_data,
            'summary': summary_data
        }

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict(data)

        return dataset
    
    def get_datasets(self):
        dataset_dict = {}

        for name, path in self.path_dict.items():
            text_data, summary_data = self.build_path_lists(path)
            dataset_dict[name] = self.build_dataset(text_data, summary_data)

        return dataset_dict
