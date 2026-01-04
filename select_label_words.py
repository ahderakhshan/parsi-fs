import argparse
import logging
import json
import os

from LabelWordExtension.label_word_extension import LabelWordExtension
from LabelWordExtension.data_processors import ParsiNLUFoodSentimentProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, KnowledgeableVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
from LabelWordExtension.contextualize_calibration import calibrate
from LabelWordExtension.filter_method import tfidf_filter
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForMaskedLM
from openprompt.plms import ModelClass, _MODEL_CLASSES, MLMTokenizerWrapper

_MODEL_CLASSES['xlmroberta'] = ModelClass(**{
    'config': XLMRobertaConfig,
    'tokenizer': XLMRobertaTokenizer,
    'model': XLMRobertaForMaskedLM,
    'wrapper': MLMTokenizerWrapper,
})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """برای استفاده از روش prompt based fine tunning یک مدل زبانی نیاز داریم که برای هر کلاس تعدادی کلمه برچسب داشته باشیم.
در ادامه نام تعدادی کلاس نمایش داده شده است. به ازای هر کدام از این کلاس ها حداکثر 30 کلمه مرتبط تولید کن. قالب خروجی باید به شکل زیر باشد. و هیچ توضیح اضافه ای قابل قبول نیست.
نام کلاس 1: کلمه برچسب 1 - کلمه برچسب 2 - کلمه برچسب 3 - ...
نام کلاس 2: کلمه برچسب 1 - کلمه برچسب 2 - کلمه برچسب  - ..."""


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--task", type=str, default="parsinlu-food-sentiment",
                                 help="must choose one of the allowed tasks")
        self.parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                                 help="pretrained language model name or path")
        self.parser.add_argument("--model_type", type=str, default='bert',
                                 help="model type which can be bert roberta xlmroberta")
        self.parser.add_argument("--filters", type=str, default=["FR", "RR"],
                                 help="Frequency Refinement and Relevance Refinement methods")
        self.parser.add_argument("--initial_label_words", type=json.loads,
                                 help="mapp label to label words. It is a dictionary which map each label"
                                      " to a label word")
        self.parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini",
                                 help="determine the version of gpt model which is used to extend label words")
        self.parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                                 help="prompt which sends to gpt model to extend label words")
        self.parser.add_argument("--template_path", type=str,
                                 default="./templates/parsinlufood/templates.txt",
                                 help="a path to template file. must defined in a way that KPT defined")
        self.parser.add_argument("--template_id", type=int, default=0,
                                 help="line number for selected template")
        self.parser.add_argument("--gpt_api_key", type=str, help="your api key to call gpt")
        self.parser.add_argument("--cutoff", type=float, default=0.5, help="threshold for the frequency refinement")
        self.parser.add_argument("--gpt_label_word_path", type=str, default="./label_words/parsinlufoodsentiemnt/gpt.txt",
                                 help="label words find with gpt are stored in this file")
        self.parser.add_argument("--final_label_word_path", type=str, default="./label_words/parsinlufoodsentiemnt/refinements.txt",
                                 help="path to label words file after final refinements")


    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_type, args.model_name_or_path)

    dataset = {}
    task = args.task
    if task == "parsinlu-food-sentiment":
        dataset['train'] = ParsiNLUFoodSentimentProcessor().get_train_examples("./data/parsinlufood/")
        dataset['test'] = ParsiNLUFoodSentimentProcessor().get_test_examples("./data/parsinlufood/")
        class_labels = ParsiNLUFoodSentimentProcessor().get_labels()
        max_seq_l = 256
        batch_s = 30

    dir_name = os.path.dirname(args.final_label_word_path)
    os.makedirs(dir_name, exist_ok=True)
    dir_name = os.path.dirname(args.gpt_label_word_path)
    os.makedirs(dir_name, exist_ok=True)

    label_word_extension = LabelWordExtension(initial_label_words=args.initial_label_words, prompt=args.prompt,
                                              model=args.gpt_type, api_key=args.gpt_api_key,
                                              output_path=args.gpt_label_word_path)

    extended_label_words = label_word_extension.extend_label_words()
    logger.info(f"Extended label words find with GPT model are: {extended_label_words}")

    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(args.template_path, choice=args.template_id)
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=args.cutoff,
                                           max_token_split=args.max_token_split).from_file(args.gpt_label_word_path)

    support_dataset = dataset['test']
    for example in support_dataset:
        example.label = -1
    support_dataloader = PromptDataLoader(dataset=support_dataset, template=mytemplate, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                          decoder_max_length=3, batch_size=batch_s, shuffle=False,
                                          teacher_forcing=False, predict_eos_token=False, truncate_method="tail")

    use_cuda = True
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                           plm_eval_mode=args.plm_eval_mode)
    if use_cuda:
        prompt_model = prompt_model.cuda()

    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    cc_logits = calibrate(prompt_model, support_dataloader)
    if "FR" in args.filters:
        myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
        logger.info(f"label words after frequency refinements are: {myverbalizer.label_words}")
    if "RR" in args.filters:
        record = tfidf_filter(myverbalizer, cc_logits, class_labels)
        logger.info(f"label words after relevance refinements are: {myverbalizer.label_words}")

    final_file = open(args.final_label_word_path, "w")
    final_file.write(myverbalizer.label_words)

    logger.info(f"label words written in the file: {args.final_label_word_path}")

