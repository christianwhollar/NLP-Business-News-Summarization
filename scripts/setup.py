import os
from scripts.build_features import BuildFeatures
from scripts.extractive_model import ExtractiveModel
from scripts.clear_data import FileDeleter
from scripts.generative_model import GenerativeModel
from  scripts.make_dataset import MakeDataset

if __name__ == '__main__':
    # File Deleter
    # fd = FileDeleter()
    # fd.delete_all_in_directory()

    # Make Data Dirs 
    if not os.path.exists('data/processed'):
        os.mkdir('data/processed')
    if not os.path.exists('data/raw'):
        os.mkdir('data/raw')

    # Download Kaggle Data
    MakeDataset()

    # Create HuggingFace Datasets
    bf = BuildFeatures()
    datasets = bf.get_datasets()

    # Gen Model Settings
    int_setting = 1

    # Model Choice
    pt_model_options = ["t5-small", "sshleifer/distilbart-cnn-12-6"]
    pt_model_name = pt_model_options[int_setting]

    # Name Choice
    abv_name_options = ["t5", "bart"]
    abv_name = abv_name_options[int_setting]
    model_save_name = f'{abv_name}_model'
    train_eval_save_name = f'{abv_name}_train_eval'
    test_eval_save_name = f'{abv_name}_test_eval'

    # Train/Test Gen Model
    print('Generative Model Training...')
    gm = GenerativeModel(datasets = datasets, model_name = pt_model_name)
    trainer = gm.get_trainer()
    gm.train_model(trainer, model_save_name = model_save_name, eval_save_name = train_eval_save_name)
    gm.test(model_name = model_save_name, eval_save_name = test_eval_save_name)
    print('Generative Model Complete!')

    # Train/Test Ext Model
    print('Extractive Model start...')
    em = ExtractiveModel(datasets=datasets)
    em.test('textrank_test_eval')
    print('Extractive Model Complete!')