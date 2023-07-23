import evaluate
import json
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

class GenerativeModel():
    '''
    
    '''

    def __init__(self, datasets, model_name = "t5-small"):
        self.datasets = datasets
        self.checkpoint = model_name

    def get_tokenizer(self, name):
        tokenizer = AutoTokenizer.from_pretrained(name)
        return tokenizer

    def get_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        return model
    
    def get_data_collator(self, tokenizer):
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = self.checkpoint)
        return data_collator
    
    def preprocess_function(self, input):
        '''
        
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
        
        '''
        tokenized_train_datasets = self.datasets['train'].map(self.preprocess_function, batched=True)
        tokenized_val_datasets = self.datasets['val'].map(self.preprocess_function, batched=True)
        return tokenized_train_datasets, tokenized_val_datasets
    
    def get_tokenized_datasets_test(self):
        '''

        '''
        tokenized_test_datasets = self.datasets['test'].map(self.preprocess_function, batched=True)
        return tokenized_test_datasets
    
    def compute_metrics(self, eval_pred):
        '''
        
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
        )

        return training_args
    
    def get_trainer(self):
        '''
        
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
        trainer.train()
        self.trainer = trainer

        evaluate = self.trainer.evaluate()

        self.trainer.save_model(f'models/{model_save_name}')

        with open(f'evals/{eval_save_name}.json', 'w') as file:
            json.dump(evaluate, file)   
    
    def generate_summary(self, model_name, text):
        '''
        
        '''
        tokenizer = self.get_tokenizer(f'models/{model_name}')
        model = AutoModelForSeq2SeqLM.from_pretrained(f"models/{model_name}")
        inputs = tokenizer(text, return_tensors="pt").input_ids
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def test(self, model_name, eval_save_name):
        '''
        
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