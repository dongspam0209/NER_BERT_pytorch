from datasets import load_dataset
import pandas as pd
wnut = load_dataset("wnut_17")
counts={idx:0 for idx in range(13)}
for idx in range(len(wnut["train"])):
    for i in range(len(wnut["train"][idx]["ner_tags"])):
        for j in range(len(counts)):
            if j==wnut["train"][idx]["ner_tags"][i]:
                counts[j]+=1
print(counts)