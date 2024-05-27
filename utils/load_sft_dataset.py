import json
import torch
import gc

from datasets import Dataset
from torch.utils.data import Dataset
from schema_item_filter_null import SchemaItemClassifierInference, filter_schema
from utils.db_utils_null import get_db_schema_sequence, get_matched_content_sequence
import json
import torch

from datasets import Dataset
from torch.utils.data import Dataset 
from transformers import AutoTokenizer
def prepare_inputs_new_format(sample, tokenizer: AutoTokenizer, max_len=8192):

    conversation = [{"role": "user", "content": sample}]
    
    input_ids = tokenizer.apply_chat_template(  
                                                conversation,
                                                tokenize=False
                                            )
    input_ids = tokenizer(input_ids).input_ids
    return {
        "input_ids": torch.tensor(input_ids[:max_len], dtype = torch.int64),
        "attention_mask": torch.tensor([1]*len(input_ids[:max_len]), dtype = torch.int64)
    }
def prepare_inputs_yi_format(sample, tokenizer: AutoTokenizer, max_len=8192):

    conversation = [{"role": "user", "content": sample}]

    input_ids = tokenizer.apply_chat_template(
                                                conversation,
                                                tokenize=True,
                                                add_generation_prompt=True,
                                                return_dict=False,
                                            )
    return {
        "input_ids": torch.tensor(input_ids[:max_len], dtype = torch.int64),
        "attention_mask": torch.tensor([1]*len(input_ids[:max_len]), dtype = torch.int64)
    }

def prepare_text2sql_prefix_sequence(data):
    #print("Schema:\n", data['schema_sequence'][:100], "\n Content: \n", data['content_sequence'])
    prefix_seq = data["schema_sequence"] + "\n" + data["content_sequence"] + "\n" + data["text"] + "\n"
    
    return prefix_seq

def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation = False)["input_ids"] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
        #print("Input ids ????", prefix_ids)
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labelsii
        labels = labels[-max_tokens:]
        
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }
def prepare_gemma_inputs(prefix_seq, tokenizer, max_prefix_length):
     
    prefix_seq = f"""<bos><start_of_turn>user
    {prefix_seq}<end_of_turn>
    <start_of_turn>model"""
    input_ids = tokenizer(prefix_seq).input_ids    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }

def prepare_llama3_inputs(prefix_seq, tokenizer, max_prefix_length):
     
    #llama3
    chat = [
    {"role": "user", "content": prefix_seq}
    ]
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True)    
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }
def prepare_alpaca(sample, tokenizer: AutoTokenizer, max_len=8192, mode="starcoder"):
    """ print("???", sample.keys())  """
    if mode == "gemma":
        return prepare_gemma_inputs(sample['instruction'], tokenizer, max_len)
    conversation = [{"role": "user", "content": sample["instruction"]}]
    
    input_ids = tokenizer.apply_chat_template(  
                                                conversation,
                                                tokenize=True,
                                                add_generation_prompt=False,
                                                return_dict=False,
                                            )
    return {
        "input_ids": torch.tensor(input_ids[:max_len], dtype = torch.int64),
        "attention_mask": torch.tensor([1]*len(input_ids[:max_len]), dtype = torch.int64)
    }
class AlpacaDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode='starcoder'):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.mode = mode
    def __getitem__(self, index):
        return prepare_alpaca(sample=self.dataset[index], tokenizer=self.tokenizer, max_len=self.max_tokens, mode=self.mode)

    def __len__(self):
        return len(self.dataset)
class SFTSQLGenerationDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        print("apply filtering strategies...")
        if mode in ["train", "debug"]:
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
        elif mode in ["eval", "dev"]:
            sic = SchemaItemClassifierInference(sic_path) 
            dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()

        # prepare schema sequence and content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])
            data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2sql_prefix_sequence(data)
        if index < 2:
            print(prefix_seq)

        if self.mode in ["train", "dev"]:
            target_seq = data["sql"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode in ["eval", "debug"]:
            return prepare_llama3_inputs(prefix_seq, self.tokenizer, self.max_tokens)

    def __len__(self):
        return len(self.dataset)

class GemmaSQLGenerationDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        print("apply filtering strategies...")
        if mode in ["train", "debug"]:
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
        elif mode in ["eval", "dev"]:
            sic = SchemaItemClassifierInference(sic_path)
            dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()

        # prepare schema sequence and content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])
            data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2sql_prefix_sequence(data)
        if index < 2:
            print(prefix_seq)

        if self.mode in ["train", "dev"]:
            target_seq = data["sql"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode in ["eval", "debug"]:
            return prepare_gemma_inputs(prefix_seq, self.tokenizer, self.max_tokens)
            """ return prepare_inputs_yi_format(prefix_seq, self.tokenizer, self.max_tokens) """

    def __len__(self):
        return len(self.dataset)

class NewFormatDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        print("apply filtering strategies...")
        if mode in ["train", "debug"]:
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
            sic = SchemaItemClassifierInference(sic_path)
            dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()

        # prepare schema sequence and content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])
            data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2sql_prefix_sequence(data)
        if index < 2:
            print(prefix_seq)

        if self.mode in ["train", "dev"]:
            target_seq = data["sql"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode in ["eval", "debug"]:
            target_seq = data["sql"]
            return prepare_inputs_new_format(prefix_seq, self.tokenizer, self.max_tokens)

        """ return prepare_inputs_yi_format(prefix_seq, self.tokenizer, self.max_tokens) """

    def __len__(self):
        return len(self.dataset)


class SQLCoderDataset(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        #return prepare_tensor_inputs(self.dataset[index], self.tokenizer, self.max_tokens)
        return prepare_llama3_inputs(self.dataset[index]['instruction'], self.tokenizer, self.max_tokens)

    def __len__(self):
        return len(self.dataset)
