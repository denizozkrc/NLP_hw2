import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model1 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)
model2 = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)

model_gpt2 = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

metric = evaluate.load("accuracy")
# TODO: max_length


def tokenize_function_gpt2(example):
    return tokenizer_gpt2(example["text"], padding="max_length", truncation=True)


def execute_gpt2(org_lang: bool, file_name: str):
    i=0
    # Load the raw test dataset
    dataset_raw_test = open(file=file_name, mode="r").read()
    dataset_dict_test = {"text": [], "label": []}

    for line in dataset_raw_test.split("\n"):
        i += 1
        if i == 100:
            break
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if parts[-1] == "label":
            continue
        text_org = parts[-3]
        text_eng = parts[-2]
        dataset_dict_test["text"].append(text_org) if org_lang else dataset_dict_test["text"].append(text_eng)
        dataset_dict_test["label"].append(int(parts[-1]))
    dataset_test = Dataset.from_dict(dataset_dict_test)
    dataset_test = dataset_test.map(tokenize_function_gpt2, batched=True)

    classifier = pipeline("text-classification", model=model_gpt2, tokenizer=tokenizer_gpt2)
    predictions = classifier(dataset_test["text"])

    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append(int(prediction["label"]))

    accuracy = 0
    for true_l, predicted_l in zip(dataset_test["label"], predicted_labels):
        if true_l == predicted_l:
            accuracy += 1
    accuracy = accuracy / len(predicted_labels)
    print("Accuracy: ", accuracy)


def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
    # tokenizer(text, return_tensors="pt", padding=True, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def execute_task1():
    # on original text
    i=0
    dataset_raw = (open(file="./datasets/orientation-tr-train.tsv", mode="r")).read()
    dataset_dict = {"text": [], "label": []}
    for line in dataset_raw.split("\n"):
        i += 1
        if i == 100:
            break
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        text_org = parts[-3]
        # text_eng = parts[-2]
        dataset_dict["text"].append(text_org)
        dataset_dict["label"].append(int(parts[-1]))
    dataset = Dataset.from_dict(dataset_dict)

    # on original text
    i=0
    dataset_raw_test = (open(file="./datasets/orientation-tr-test.tsv", mode="r")).read()
    dataset_dict_test = {"text": [], "label": []}
    for line in dataset_raw_test.split("\n"):
        i += 1
        if i ==100:
            break;
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        print(parts)
        text_org = parts[-3]
        # text_eng = parts[-2]
        dataset_dict_test["text"].append(text_org)
        dataset_dict_test["label"].append(int(parts[-1]))
    dataset_test = Dataset.from_dict(dataset_dict_test)
    #ds1 = dataset1

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
    )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset_test = dataset_test.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model1,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results task1:", evaluation_results)

    execute_gpt2(True, "./datasets/orientation-tr-test.tsv")  # on original lang (tr)
    execute_gpt2(False, "./datasets/orientation-tr-test.tsv")  # on english


def execute_task2():
    # on english translated text 
    i=0
    dataset_raw = (open(file="./datasets/power-tr-train.tsv", mode="r")).read()
    dataset_dict = {"text": [], "label": []}
    for line in dataset_raw.split("\n"):
        i += 1
        if i ==100:
            break;
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        # text_org = parts[-3]
        text_eng = parts[-2]
        dataset_dict["text"].append(text_eng)
        dataset_dict["label"].append(int(parts[-1]))
    dataset = Dataset.from_dict(dataset_dict)

    # on english translated text 
    i=0
    dataset_raw_test = (open(file="./datasets/power-tr-test.tsv", mode="r")).read()
    dataset_dict_test = {"text": [], "label": []}
    for line in dataset_raw_test.split("\n"):
        i += 1
        if i ==100:
            break;
        if len(line) == 0:
            continue
        parts = line.split("\t")
        if (parts[-1] == "label"):
            continue
        # text_org = parts[-3]
        text_eng = parts[-2]
        dataset_dict_test["text"].append(text_eng)
        dataset_dict_test["label"].append(int(parts[-1]))
    dataset_test = Dataset.from_dict(dataset_dict_test)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
    )

    dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_results = trainer.evaluate()
    print("Evaluation Results task2: ", evaluation_results)

    execute_gpt2(True, "./datasets/power-tr-test.tsv")  # on original lang (tr)
    execute_gpt2(False, "./datasets/power-tr-test.tsv")  # on english


# TODO: Stratified k-fold 1 to 9 for the shared task

# Tokenization check
#for i in range(len(dataset1)):
#    if(dataset1[i]["label"] != ds1[i]["label"]):
#        print("Error in tokenization")

execute_task1()
# execute_task2()
