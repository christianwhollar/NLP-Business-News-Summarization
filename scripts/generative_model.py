import evaluate
import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

class GenerativeModel():
    '''
    A class representing a Generative Model for text summarization using HuggingFace's Transformers.

    Attributes:
        datasets (Dataset): HuggingFace datasets containing train, val, and test data.
        checkpoint (str): Name of the pre-trained model used for summarization.

    Methods:
        __init__(self, datasets, model_name="t5-small"): Initializes the GenerativeModel with datasets and model_name.
        get_tokenizer(self, name): Returns the tokenizer for the specified pre-trained model.
        get_model(self): Returns the Seq2SeqLM model for text summarization.
        get_data_collator(self, tokenizer): Returns the data collator for Seq2Seq models.
        preprocess_function(self, input): Preprocesses the input data for training the model.
        get_tokenized_datasets_train(self): Returns tokenized training and validation datasets.
        get_tokenized_datasets_test(self): Returns tokenized test dataset.
        compute_metrics(self, eval_pred): Computes evaluation metrics (ROUGE) for model performance.
        get_training_args(output_dir="models/generative"): Returns training arguments for the model.
        get_trainer(self): Returns the Seq2SeqTrainer for training the model.
        train_model(self, trainer, model_save_name, eval_save_name): Trains the model and saves the results.
        generate_summary(self, model_name, text): Generates a summary for the given input text.
        test(self, model_name, eval_save_name): Performs model testing and saves evaluation results.
    '''
    
    def __init__(self, datasets, model_name = "t5-small"):
        '''
        Returns the tokenizer for the specified pre-trained model.

        Args:
            name (str): Name of the pre-trained model.

        Returns:
            tokenizer: The tokenizer for the specified model.
        '''
        self.datasets = datasets
        self.checkpoint = model_name

    def get_tokenizer(self, name):
        '''
        Returns the tokenizer for the specified pre-trained model.

        Args:
            name (str): Name of the pre-trained model.

        Returns:
            tokenizer: The tokenizer for the specified model.
        '''
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer

    def get_model(self):
        '''
        Returns the Seq2SeqLM model for text summarization.

        Returns:
            model: The Seq2SeqLM model.
        '''
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        return model
    
    def get_data_collator(self, tokenizer):
        '''
        Returns the data collator for Seq2Seq models.

        Args:
            tokenizer: The tokenizer for the specified model.

        Returns:
            data_collator: The data collator for Seq2Seq models.
        '''
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = self.checkpoint)
        return data_collator
    
    def preprocess_function(self, input):
        '''
        Preprocesses the input data for training the model.

        Args:
            input: The input data containing text and summary.

        Returns:
            model_inputs: Preprocessed data as model inputs.
        '''

        tokenizer = self.get_tokenizer(self.checkpoint)
        prefix = "summarize: "
    
        inputs = [prefix + doc for doc in input["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=input["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_tokenized_datasets_train(self):
        '''
        Returns tokenized training and validation datasets.

        Returns:
            tokenized_train_datasets, tokenized_val_datasets: A tuple containing tokenized training and validation datasets.
        '''
        tokenized_train_datasets = self.datasets['train'].map(self.preprocess_function, batched=True)
        tokenized_val_datasets = self.datasets['val'].map(self.preprocess_function, batched=True)
        return tokenized_train_datasets, tokenized_val_datasets
    
    def get_tokenized_datasets_test(self):
        '''
        Returns tokenized test dataset.

        Returns:
            tokenized_test_datasets: Tokenized test dataset.
        '''
        tokenized_test_datasets = self.datasets['test'].map(self.preprocess_function, batched=True)
        return tokenized_test_datasets
    
    def compute_metrics(self, eval_pred):
        '''
        Computes evaluation metrics (ROUGE) for model performance.

        Args:
            eval_pred: The evaluation predictions.

        Returns:
            dict: A dictionary containing computed evaluation metrics.
        '''
        rouge = evaluate.load('rouge')
        tokenizer = self.get_tokenizer(self.checkpoint)
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    def get_training_args(output_dir = "models/generative"):
        '''
        Returns training arguments for the model.

        Args:
            output_dir (str): Directory to save the model to during training.

        Returns:
           training_args (Seq2SeqTrainingArguments): Training arguments for the model.
        '''
        training_args = Seq2SeqTrainingArguments(
            output_dir="models/my_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=10,
            predict_with_generate=True,
            fp16=True
        )

        return training_args
    
    def get_trainer(self):
        '''
        Returns the Seq2SeqTrainer for training the model.

        Returns:
            trainer (Seq2SeqTrainer): The trainer for the Seq2Seq model.
        '''
        model = self.get_model()
        training_args = self.get_training_args()
        tokenized_train_dataset, tokenized_val_dataset = self.get_tokenized_datasets_train()
        tokenizer = self.get_tokenizer(self.checkpoint)
        data_collator = self.get_data_collator(tokenizer=tokenizer)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        return trainer
    
    def train_model(self, trainer, model_save_name, eval_save_name):
        '''
        Trains the model and saves the results.

        Args:
            trainer (Seq2SeqTrainer): The trainer for the Seq2Seq model.
            model_save_name (str): Name to save model file as.
            eval_save_name (str): Name to save evaluation file as.
        '''
        trainer.train()
        self.trainer = trainer

        evaluate = self.trainer.evaluate()

        self.trainer.save_model(f'models/{model_save_name}')

        with open(f'evals/{eval_save_name}.json', 'w') as file:
            json.dump(evaluate, file)   
    
    def generate_summary(self, model_name, text):
        '''
        Generates a summary for the given input text.

        Args:
            model_name (str): Name of the pre-trained model to use.
            text (str): Input text for summarization.

        Returns:
            summary(str): The generated summary.
        '''
        tokenizer = self.get_tokenizer(f'models/{model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained(f"models/{model_name}")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding= "max_length").input_ids
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def test(self, model_name, eval_save_name):
        '''
        Performs model testing and saves evaluation results.

        Args:
            model_name (str): Name of the pre-trained model to use for testing.
            eval_save_name (str): Name to evaluation file as.
        '''
        rouge = evaluate.load("rouge")
        refs = []
        summaries = []

        for test_data in self.datasets['test']:
            pred_summary = self.generate_summary(model_name, test_data['text'])
            refs.append(test_data['summary'])
            summaries.append(pred_summary)

        rouge_scores = rouge.compute(predictions=summaries, references=refs)
        
        with open(f'evals/{eval_save_name}.json', 'w') as file:
            json.dump(rouge_scores, file)