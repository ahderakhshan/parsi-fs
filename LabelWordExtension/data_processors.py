from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor

import os
import csv
import pandas as pd
from openprompt.data_utils.utils import InputExample


class ParsiNLUFoodSentimentProcessor(DataProcessor):
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


