import pandas as pd
from collections import defaultdict
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

for idx in range(10):
    print(dataset[idx])