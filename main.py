import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, AutoModel
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
tokenizer_eng = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model_llama = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B", num_labels=2
)


# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)

model_gpt2 = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
# model_gpt2 = GPT2Model.from_pretrained("gpt2", num_labels=2)

metric = evaluate.load("accuracy")


def execute_llama(dataset_test):
    # Use the LLaMA model with a zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model=model_llama, tokenizer=tokenizer_llama, device =-1)

    valid_texts = [text for text in dataset_test["text"] if text and text.strip()]
    valid_labels = [dataset_test["label"][i] for i, text in enumerate(dataset_test["text"]) if text and text.strip()]
    predictions = classifier(valid_texts, [1, 0]) # Pass the filtered texts

    #predictions = classifier(dataset_test["text"])

    print(predictions[0])

    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append(int(prediction["label"][-1]))  # 1 or 0

    accuracy = 0
    for true_l, predicted_l in zip(dataset_test["label"], predicted_labels):
        if true_l == predicted_l:
            accuracy += 1
    accuracy = accuracy / len(predicted_labels)
    return (accuracy)


def execute_gpt2(dataset_test):
    classifier = pipeline("zero-shot-classification", model=model_gpt2, tokenizer=tokenizer_gpt2)
    # classifier = pipeline("text-classification", model=model_gpt2, tokenizer=tokenizer_gpt2, device=0)
    predictions = classifier(dataset_test["text"])
    # inputs = tokenizer_gpt2(dataset_test["text"], padding=True, truncation=True, return_tensors="pt")
    # outputs = model_gpt2(**inputs)

    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append(int(prediction["label"][-1]))

    accuracy = 0
    for true_l, predicted_l in zip(dataset_test["label"], predicted_labels):
        if true_l == predicted_l:
            accuracy += 1
    accuracy = accuracy / len(predicted_labels)
    return (accuracy)


def tokenize_function(example):
    return tokenizer(example["text"], padding=True, truncation=True)
    # tokenizer(text, return_tensors="pt", padding=True, truncation=True)


def tokenize_function_eng(example):
    return tokenizer_eng(example["text"], padding=True, truncation=True)
    # tokenizer(text, return_tensors="pt", padding=True, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def execute_task(org_lang: bool, is_orientation: bool):
    dataset_raw = pd.read_csv("./orientation-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"]) if is_orientation else pd.read_csv("./power-tr-train.tsv", sep="\t", header=None, names=["id", "speaker", "sex", "text_org", "text_eng", "label"])
    dataset = dataset_raw[dataset_raw["label"] != "label"]
    dataset = dataset.iloc[:10] # for trial phase
    dataset = dataset[dataset["text_org"].notnull()]
    dataset["label"] = dataset["label"].astype(int)

    dataset_train_split, dataset_test_split = train_test_split(dataset, test_size=0.1, random_state=0) # TODO: random state

    dataset_train_dict = {
        "text": dataset_train_split["text_org"].tolist() if org_lang else dataset_train_split["text_eng"].tolist(),
        "label": dataset_train_split["label"].tolist()
    }

    dataset_test_dict = {
        "text": dataset_test_split["text_org"].tolist() if org_lang else dataset_test_split["text_eng"].tolist(),
        "label": dataset_test_split["label"].tolist()
    }

    dataset_train = Dataset.from_dict(dataset_train_dict)
    dataset_test = Dataset.from_dict(dataset_test_dict)

    """training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=500,
        #evaluation_strategy="steps",
        #eval_steps=10,
        num_train_epochs=1,
        #load_best_model_at_end=True,
        report_to=["tensorboard"],
        logging_dir="./logs",
        per_device_train_batch_size=8,  # Reduce if memory is an issue
        save_steps=10,
        save_total_limit=2,
    )"""

    dataset_train_mapped = dataset_train.map(tokenize_function, batched=True) if org_lang else dataset_train.map(tokenize_function_eng, batched=True)
    dataset_test_mapped = dataset_test.map(tokenize_function, batched=True) if org_lang else dataset_test.map(tokenize_function_eng, batched=True)

    """trainer = Trainer(
        model=model1,
        args=training_args,
        train_dataset=dataset_train_mapped,
        eval_dataset=dataset_test_mapped,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results original language:", evaluation_results) if org_lang else print("Evaluation Results English:", evaluation_results)
"""
    """accuracy = execute_gpt2(dataset_test_mapped)  # on original lang (tr)
    print("original language, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)

    accuracy = execute_gpt2(dataset_test_mapped)  # on english
    print("english, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)"""

    accuracy = execute_llama(dataset_test_mapped)  # on original lang (tr)
    print("original language, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)

    accuracy = execute_llama(dataset_test_mapped)  # on english
    print("english, ", "is orientation: ", is_orientation)
    print("Accuracy: ", accuracy)


# TODO: Stratified k-fold 1 to 9 for the shared task

# Tokenization check
#for i in range(len(dataset1)):
#    if(dataset1[i]["label"] != ds1[i]["label"]):
#        print("Error in tokenization")

execute_task(True, True)
execute_task(False, False)
