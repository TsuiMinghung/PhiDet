# this module reads data from csv and forms two datasets extend pytorch
# util.data called test.pt and train.pt

import json
from pandas import read_excel
from phish_data import PhishEmail
from phish_data import PhishDataset
import torch

samples = []
df = read_excel('/home/alex/study/gra-pro/PhiDet/dataset/answer2.xlsx')
label_map = dict(zip(df['email_id'],df['spam_type']))

with open("/home/alex/study/gra-pro/PhiDet/dataset/spam_email_data.log") as file:
    for line in file:
        id, json_string = line.split("\t")[0::1]
        parsed_json = json.loads(json_string)
        from_name = parsed_json['fromname']
        subject = parsed_json['subject']
        content = parsed_json['content']
        label = label_map[id]
        samples.append(PhishEmail(subject,content,from_name,label))

train_samples , test_samples = samples[:int(len(samples) * 0.8)] , samples[int(len(samples) * 0.8):]

train_dataset = PhishDataset(train_samples)
test_dataset = PhishDataset(test_samples)

torch.save(train_dataset,'./train.pt')
torch.save(test_dataset,'./test.pt')
