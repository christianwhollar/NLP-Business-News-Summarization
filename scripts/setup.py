from scripts.build_features import BuildFeatures
from scripts.extractive_model import ExtractiveModel
from  scripts.make_dataset import MakeDataset

if __name__ == '__main__':
    MakeDataset()

    bf = BuildFeatures()
    datasets = bf.get_datasets()

    # em = ExtractiveModel(datasets=datasets)
    # em.test('TextRankScores')