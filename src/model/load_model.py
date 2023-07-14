from transformers import BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

from .model import E_BERT

def get_model():

    model_config = BertConfig.from_pretrained('model/kmbert_vocab/bert_config.json')
    model_config.num_labels = 4
    model_config.problem_type = 'single_label_classification'
    base_model = E_BERT.from_pretrained('model/kmbert_vocab/pytorch_model.bin',
                                        config=model_config)
    
    return base_model

def get_tokenizer():

    tokenizer = BertTokenizer('model/kmbert_vocab/kmbert_vocab.txt', do_lower_case=False)
    
    return tokenizer