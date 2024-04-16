from transformers import AutoTokenizer,DataCollatorForTokenClassification
import pandas as pd
import numpy as np
import torch
import re
# load dataset
df=pd.read_csv('./BIO-FallRisk_ko_No2_10000_YGB-ano2.csv',encoding='utf-8')
label_list=['O',
            'B-Beh_Careless',
            'I-Beh_Careless',
            'B-Beh_Non_cooper',
            'I-Beh_Non_cooper',
            'B-Beh_Overconfidence',
            'I-Beh_Overconfidence',
            'B-Cog_Confusion',
            'I-Cog_Confusion',
            'B-Cog_Delirium',
            'I-Cog_Delirium',
            'B-Cog_Dementia',
            'B-Cog_DisOT',
            'I-Cog_DisOT',
            'B-Cog_Excitment',
            'I-Cog_Excitment',
            'B-Cog_LOC',
            'I-Cog_LOC',
            'B-Mob_Aids',
            'I-Mob_Aids',
            'B-Mob_Dizz',
            'B-Mob_P_limit',
            'I-Mob_P_limit',
            'B-Mob_Weak',
            'I-Mob_Weak',
            'B-Sen_Hearing_Imp',
            'I-Sen_Hearing_Imp',
            'B-Sen_Visual_Imp',
            'I-Sen_Visual_Imp',
            'B-Slp_Imp',
            'I-Slp_Imp',
            'B-Slp_Sedatives',
            'I-Slp_Sedatives',
            'B-Toil_Urgency',
            'I-Toil_Urgency',
            'B-Tx_Restraint',
            'B-Tx_RiskMed',
            'I-Tx_RiskMed',
            'B-Tx_RiskPro',
            'I-Tx_RiskPro']
label2id={tag:idx for idx,tag in enumerate(label_list)}
id2label={value:key for key,value in label2id.items()}

dataset=[]
grouped_data_tokens=df.groupby('Sentence #')['token'].apply(list).reset_index()
grouped_data_tags=df.groupby('Sentence #')['tag'].apply(list).reset_index()
for idx in range(len(grouped_data_tokens)):
    temp_dict={
        'id':idx,
        'tokens':grouped_data_tokens['token'][idx],
        'ner_tags':[label2id.get(tag, 0) for tag in grouped_data_tags['tag'][idx]]
    }
    dataset.append(temp_dict)

np.random.seed(99)
np.random.shuffle(dataset)
train_ratio=int(len(dataset)*0.75)

train_dataset=dataset[:train_ratio]
test_dataset=dataset[train_ratio:]

# to make the dataset in format like below
# {'id': '0',
#  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
#  'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
# }

tokenizer=AutoTokenizer.from_pretrained('klue/bert-base')    

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example['tokens'], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(example['ner_tags'][word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs['labels'] = label_ids
    return tokenized_inputs

# tokenize dataset
tokenized_train_dataset = [tokenize_and_align_labels(example) for example in train_dataset]
tokenized_test_dataset = [tokenize_and_align_labels(example) for example in test_dataset]

# tokenized dataset check

#create batch of examples
data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)

#evaluate
import evaluate
seqeval=evaluate.load("seqeval")
example=dataset[0]
labels=[label_list[i]for i in example[f"ner_tags"]]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    "klue/bert-base", num_labels=40, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="klubert_Fallrisk_NER",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()