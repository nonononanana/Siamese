import pandas as pd
import numpy as np
import yaml
import re


class DataExtractor(object):
    def __init__(self, config):


        if config["data"]["training_dataset"] == "quora":
            train_data = pd.read_csv(config["data"]["training_path"])
            np.random.seed(42)
            mask = np.random.rand(len(train_data)) < config["data"]["train_val_split"]
            self.training_data = train_data[mask]
            self.val_data = train_data[~mask]

            self.q1_index = 'question1'
            self.q2_index = 'question2'
            self.label_index = 'is_duplicate'

        if config["data"]["testing_dataset"] == "quora":
            self.test_data = pd.read_csv(config["data"]["testing_path"])

            self.q1_index_test = 'question1'
            self.q2_index_test = 'question2'

        # print ( training_data, val_data, test_data)

    def get_training_list(self):
        # return 3 lists of question1, question2, label
        q1 = self.training_data[self.q1_index].tolist()
        q2 = self.training_data[self.q2_index].tolist()
        labels = self.training_data[self.label_index].tolist()
        return  [self.preprocess(q) for q in q1], [self.preprocess(q) for q in q2], labels

    def get_val_list(self):
        # return 3 lists of question1, question2, label
        q1 = self.val_data['question1'].tolist()
        q2 = self.val_data['question2'].tolist()
        labels = self.val_data['is_duplicate'].tolist()
        return  [self.preprocess(q) for q in q1], [self.preprocess(q) for q in q2], labels

    def get_testing_list(self):
        q1 = self.test_data[self.q1_index_test].tolist()
        q2 = self.test_data[self.q2_index_test].tolist()

        return [self.preprocess(q) for q in q1], [self.preprocess(q) for q in q2]



    def preprocess(self, text):
        try:
            text = text.lower()
        except:
            print (1)
            return ""

        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return (text)

with open('config.yaml', 'r') as f:
    config = yaml.load(f)


