## Guide for our evalution process
We use our single model for evaluation (StarCoder-v2 15B instruct as base model) and LLama 3 tokenizer for preprocessing data.
### Prepare All Environments
Pull docker image
```
docker pull chideptrai/bird_leaderboard
docker run --gpus $gpu_device -i -t  chideptrai/bird_leaderboard  /bin/bash
```
Download conda environment and update pyarrow (if necessary)
```
cd ~/text2sql_repo
huggingface-cli download "ambivalent02/sql_env" --local-dir ./submit_env

mv ./submit_env/submit_env.zip . && rm -rf ./submit_env

unzip submit_env.zip && rm submit_env.zip

conda activate ./submit_env

python3 -m pip install -U pyarrow
```

#### Step3: Download Checkpoints 

```
#LLM
huggingface-cli download "ambivalent02/sql_15b_vinai" \
--local-dir ./submit_model
#Classifier
huggingface-cli download "ambivalent02/roberta_sql_sic" \
--local-dir ./submit_sic_ckp
```

#### Step4: Setup and Pre-process data
```
# Build BM25 index for each 
python build_contents_index.py \
--db_root_path $path_to_test_database \
--index_path $path_to_save_test_index    

# Pre-process dataset
python prepare_sft_datasets.py \
--dataset $path_to_test_dir\
--db_root_path $path_to_test_database\
--index_path
$path_to_saved_test_index

# Add to null to prompt template
python convert_to_null_format.py \
--dataset ./submit.json \
--db_root_path $path_to_test_database \
--outfile ./submit_null.json

# Convert data to alpaca format 
python convert_sql_to_alpaca.py \
path_to_llama3_model \
"./submit_null.json"
```
### Step5: Run Inference and Evaluation
```
bash run_evaluation.sh 
"./submit_model" "starcoder" 
"./submit_null_alpaca.json" 
```
(Example data: ./dev -> replace it with test data)
