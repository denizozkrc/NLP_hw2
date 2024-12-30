# from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
# model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=2)  # Adjust `num_labels`


dataset1_raw = (open(file="./datasets/orientation-tr-train.tsv", mode="r")).read()
dataset1_inputs = []
dataset1_labels = []
for line in dataset1_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    text_org = parts[-3]
    text_eng = parts[-2]
    dataset1_inputs.append(text_org)
    dataset1_labels.append(parts[-1])
del dataset1_inputs[0]
del dataset1_labels[0]
print(dataset1_inputs[0:2])

dataset2_raw = (open(file="./datasets/power-tr-train.tsv", mode="r")).read()
dataset2_inputs = []
dataset2_labels = []
for line in dataset2_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    text_org = parts[-3]
    text_eng = parts[-2]
    dataset2_inputs.append(text_org)
    dataset2_labels.append(parts[-1])
del dataset2_inputs[0]
del dataset2_labels[0]

# TODO: Stratified k-fold 1 to 9 for the shared task

# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get model outputs
# outputs = model(**inputs)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

for i, text in enumerate(dataset1_inputs):
    input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input["labels"] = int(dataset1_labels[i])
    dataset1_inputs[i] = input

"""trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset1DS,
    # eval_dataset=eval_dataset, # use this later
)

trainer.train()"""

