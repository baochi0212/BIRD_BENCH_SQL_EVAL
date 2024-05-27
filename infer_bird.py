import argparse
import os
import torch
import json
import time
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_sft_dataset import SFTSQLGenerationDataset, NewFormatDataset, SQLCoderDataset, GemmaSQLGenerationDataset, AlpacaDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Filter warnings by category
warnings.filterwarnings("ignore")
# Log ex error file:


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)

    parser.add_argument('--dataset_path', type = str)

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)
    parser.add_argument('--chunk_len', type=int, required=True)
    parser.add_argument('--model_type', type=str, required=True) 
    opt = parser.parse_args()

    return opt

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql

def text2sql_func(model, inputs, tokenizer, max_new_tokens):
    #print("Inputs shape: ", inputs['input_ids'].shape)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, 128009]
        )
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)

    return generated_sqls

if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
       

    if opt.model_type == "starcoder" or opt.model_type == "gemma":
        raw_dataset = json.load(open(f"./bird_dev_ex_chunk_{opt.index}.json"))
        eval_set = AlpacaDataset(
                opt.dataset_path,
                tokenizer,
                max_tokens - max_new_tokens,
                opt.model_type
                )

    check_item = eval_set[0]
    """ print("Sanity check: ", tokenizer.decode(check_item['input_ids'])) """

    # TODO: current, we only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size = 1)
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, device_map = "auto", torch_dtype = torch.float16)
    # 
    model.eval()
    start_time = time.time()
    predicted_sqls = []
    for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        generated_sqls = text2sql_func(model, batch_data, tokenizer, max_new_tokens)
        """ print("----------------------Gen decode: ", generated_sqls) """
        generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in generated_sqls]
        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, raw_data["db_path"])
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql
                break
        if final_generated_sql is None:
            final_generated_sql = f"Error: {execution_error}; Pred: {generated_sql}"
        
        print("Final: ", final_generated_sql)
        predicted_sqls.append(final_generated_sql)

        
    end_time = time.time()
    print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        opt.llm_path, 
        end_time - start_time,
        len(raw_dataset),
        (end_time - start_time) / len(raw_dataset)
        )
    )


    if "bird" in opt.dataset_path:
        bird_results_dict = dict()
        for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
            bird_results_dict[idx + (int(opt.index)-1)*opt.chunk_len] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
        with open(f"predict_dev_{opt.index}.json", "w", encoding = 'utf-8') as f:
            f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))


