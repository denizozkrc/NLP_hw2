from sklearn.model_selection import StratifiedKFold
import numpy as np

dataset1_raw = (open(file="./datasets/orientation-tr-train.tsv", mode="r")).read()
dataset1 = []
for line in dataset1_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    text_org = parts[-3].split(" ")
    text_eng = parts[-2].split(" ")
    dataset1.append(parts[0:-3] + [text_org] + [text_eng] + [parts[-1]])
del dataset1[0]

dataset2_raw = (open(file="./datasets/power-tr-train.tsv", mode="r")).read()
dataset2 = []
for line in dataset2_raw.split("\n"):
    if len(line) == 0:
        continue
    parts = line.split("\t")
    text_org = parts[-3].split(" ")
    text_eng = parts[-2].split(" ")
    dataset2.append(parts[0:-3] + [text_org] + [text_eng] + [parts[-1]])
del dataset2[0]

#Stratified k-fold 1 to 9

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # TODO: delete random_state=0

skf.get_n_splits()
print(skf)


