from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor

import os
import csv
import pandas as pd
from openprompt.data_utils.utils import InputExample


class ParsiNLUSentimentProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["خوب", "بد", "متوسط"]
        self.label_column_to_ids = {"Positive": 0, "Neutral": 1, "Negative": 2}

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        dataset = pd.read_csv(path, header=None)
        for index, row in dataset.iterrows():
            label, sentence = row[0], row[1]
            example = InputExample(guid=str(index), text_a=sentence, label=self.label_column_to_ids[row[0]])
            examples.append(example)
        return examples


class ParsiNLUNLI(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["استلزام", "خنثی", "تناقض"]
        self.label_column_to_ids = {"e": 0, "c": 2, "n": 1}
        self.punctuations = ["،","؛",":","؟","!",".","—","-","%"]

    def get_train_examples(self, data_dir, replace_a_char=False, replace_b_char=False) -> InputExample:
        return self.get_examples(data_dir, "train", replace_a_char, replace_b_char)

    def get_test_examples(self, data_dir, replace_a_char=False, replace_b_char=False) -> InputExample:
        return self.get_examples(data_dir, "test", replace_a_char, replace_b_char)

    def get_examples(self, data_dir, split, replace_a_char=False, replace_b_char=False):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        dataset = pd.read_csv(path, header=None)
        for index, row in dataset.iterrows():
            label, sentence_a, sentence_b = row[2], row[0], row[1]
            if replace_a_char and (sentence_a[-1] in self.punctuations):
                sentence_a = sentence_a[0:-1]
                sentence_a += replace_b_char
            elif replace_a_char:
                sentence_a += replace_a_char

            if replace_b_char and (sentence_b[-1] in self.punctuations):
                sentence_b = sentence_b[0:-1]
                sentence_b += replace_b_char
            elif replace_b_char:
                sentence_b += replace_b_char

            example = InputExample(guid=str(index), text_a=sentence_a, text_b=sentence_b,
                                   label=self.label_column_to_ids[label])
            examples.append(example)
        return examples



