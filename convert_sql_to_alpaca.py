import sys
from transformers import AutoTokenizer
sys.path.append("/home/chitb/phogpt_path/Text2sql")
from utils.load_sft_dataset import SFTSQLGenerationDataset
import json
from tqdm import tqdm
import multiprocessing
import torch
import random
import numpy as np
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(333)
write_dict = []
max_len = 0
llama3_tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
llama3_tokenizer.pad_token_id = llama3_tokenizer.eos_token_id
#DEV SET
dataset = SFTSQLGenerationDataset(sys.argv[2], llama3_tokenizer, 8192, "dev", 6, 10, "./submit_sic_ckp/")
raw_dataset = json.load(open(sys.argv[2]))

for i in tqdm(range(len(dataset))):
    instruction = llama3_tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
    dict = ({
        "db_id": raw_dataset[i]['db_id'],
        #"instruction": f"Your task is generate the SQL query given the database schema and text input. Remember to generate only SQL query without explanation\n{instruction}SQL: ",
        "instruction": instruction,
        "input": "",
        "output": llama3_tokenizer.decode([item for item in dataset[i]['labels'] if item != -100], skip_special_tokens=True)
                      })
    dict['instruction'] = dict['instruction'].replace(dict['output'], "")
    if len(llama3_tokenizer(dict['instruction']).input_ids) > max_len:
        max_len = len(llama3_tokenizer(dict['instruction']).input_ids)
    write_dict.append(dict)

with open("./submit_null_alpaca.json", "w") as f:
    json.dump(write_dict, f, indent=3)
print("Max length: ", max_len)
