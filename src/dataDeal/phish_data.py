from torch.utils.data import Dataset
from pandas import read_excel
import json

class PhishDataset(Dataset):
    def __init__(self,csv_file,label_file,root_dir,transform = None):
        super().__init__()
        self.transform = transform
        self.samples = []
        df = read_excel(str(root_dir + csv_file))
        label_map = dict(zip(df['email_id'], df['spam_type']))

        with open(str(root_dir + label_file)) as file:
            for line in file :
                id, json_string = line.split("\t")[0 : :1]
                parsed_json = json.loads(json_string)
                from_name = parsed_json['fromname']
                subject = parsed_json['subject']
                content = parsed_json['content']
                label = label_map[id]
                self.samples.append(PhishEmail(subject, content, from_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class PhishEmail:
    def __init__(self,subject,content,from_name,label):
        self.subject = subject
        self.content = content
        self.from_name = from_name
        self.label = 1 if label == 5 else 0 # label 5 means normal , others are phishing