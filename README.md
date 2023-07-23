# NLP-Business-News-Summarization
### Project
Summarize business news articles using a deep learning approach (fine tuned text summarization transformer model) and non-deep learning approach.

Transformer Models:
* DistilBart
* Google T5

Extractive Approach:
* TextRank

Kaggle Datasets:
* BBC News Summary (pariza/bbc-news-summary)
* News Category Dataset (rmisra/news-category-dataset)

### Business News Summarization App Instructions
To just run the Streamlit App, do the following:
1. Create a new conda environment and activate it: 
    ```
    conda create --name cv python=3.10
    conda activate cv
    ```
2. Install python package requirements:*
    ```
    pip install -r requirements.txt 
    ```
3. Run the streamlit app:
    ```
    !streamlit run app/SummarizeBusinessArticle.py
    ```
#### Repository Structure
```
├── README.md                        <- description of project and how to set up and run it
├── requirements.txt                 <- requirements file to document dependencies
├── setup.py                         <- script to set up project (get data, build features, train model)
├── main.ipynb                       <- main script/notebook to run model, streamlit web application from colab
├── app                              <- directory for app and scripts
    ├── SummarizeBusinessArticle.py  <- streamlit web application to generate business news summaries
    ├── summarizer.py                <- generate / extract summary functionality
├── scripts                          <- directory for pipeline scripts or utility scripts
    ├── setup.py                     <- full model pipeline
    ├── clear_data.py                <- delete all dirs/files in data/processed/
    ├── make_dataset.py              <- orchestrate kaggle data download
    ├── build_features.py            <- build HuggingFace datasets
    ├── generative_model.py          <- script to train and test abstractive models
    ├── extractive_model.py          <- script to test extractive models
├── models                           <- directory for trained models
    ├── bart_model.py                <- trained distilbart model
    ├── t5_model.py                  <- trained t5 model
├── data                             <- directory for project data
    ├── constants.py                 <- load kaggle credentials
    ├── download_data.py             <- download kaggle datasets to /data/raw/
    ├── kaggle.json                  <- kaggle credentials file
    ├── raw                          <- directory to store raw kaggle data
    ├── processed                    <- directory to store processed kaggle data
├── evals                            <- directory for rouge scores in json format
    ├── bart_train_eval.json         <- distilbart train data rouge scores
    ├── bart_test_eval.json          <- distilbart test data rouge scores
    ├── t5_train_eval.json           <- t5 train data rouge scores
    ├── t5_test_eval.json            <- t5 test data rouge scores
    ├── textrank_test_eval.json      <- textrank test data rouge scores
├── .gitignore                       <- git ignore file
```
### Pipeline Instructions
Change int_setting in [/main/scripts/setup.py](https://github.com/christianwhollar/NLP-Business-News-Summarization/blob/main/scripts/setup.py) to 0 for **Google T5** or 1 for **DistilBart**. Running ```%run scripts/setup.py``` will do the following:
* clean the data directory
* download the datasets from Kaggle
* process the data and build HuggingFace datasets
* train the generative model (train/val datasets)
* test the generative model (test dataset)
* test the extractive model

ROUGE evaluation scores can found in [/main/evals/](https://github.com/christianwhollar/NLP-Business-News-Summarization/blob/main/evals/)

### Generative Model Instructions

Create an abstractive text summarizer using the GenerativeModel Class:

* **datasets** (HuggingFace Dataset): text feature for article texts, summary feature for article summaries
* **model_name** (str): choose model to fine tune. options are 't5_model' or 'bart_model'

```python
from scripts.generative_model import GenerativeModel
gm = GenerativeModel(datasets = datasets, model_name = pt_model_name)
```
Train the model:
```
trainer = gm.get_trainer()
gm.train_model(trainer, model_save_name = model_save_name, eval_save_name = train_eval_save_name)
```
Test the model:

* **model_save_name** (str): save name of model trained in previous step
* **eval_test_name** (str): save name of .json file containing ROUGE scores
```python
gm.test(model_name = model_save_name, eval_save_name = test_eval_save_name)
```
Summarize with the model:
```python
summary = gm.generate_summary(model_name, text)
```
* **model_name** (str): save name of model to load
* **text** (str): article body text to summarize

### Extractive Model Instructions
Create an extractive text summarizer using the ExtractiveModel Class:

* **text** (str): article body text to summarize
```python
esm = ExtSummarizer()
ext_sents = esm.get_sentences(text) # Extract sentences from article text
processed_article = esm.preprocess(ext_sents) # Preprocess sentences
fvs = esm.vectorize(processed_sents = processed_article) # Convert to feature vectors using Word Count
adj_mat = esm.generate_adjacency_matrix(feature_vecs = fvs) # Create graph data structure for relationships
summary = esm.summarize(ext_sents, adj_mat, top_n = 3) # Generate summary
```