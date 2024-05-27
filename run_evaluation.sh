#model type: codes, new_format, ...
model_type=$2
#track time:
batch_per_gpu=1
start_time=$(date +%s)
#split the input json file
#num_gpus=`nvidia-smi -L | wc -l`
num_gpus=8
echo "Num gpus: $num_gpus"
prefix="chunk_bird"
#remove chunks; predict outputs
rm -rf ${prefix}*.json
rm -rf predict*.json
rm -rf bird_dev_ex_chunk*.json
#chunking
num_chunks=$((num_gpus*batch_per_gpu)) #each A100 4 chunks
echo "Num chunks: $num_chunks"
#get the chunk length
dev_len=`python ./get_json_len.py $3`
echo "Dev len: $dev_len"
chunk_len=$(($dev_len/$num_chunks  + 1))
echo "Chunk length: $chunk_len"
python ./split_chunk_json.py $3 $prefix $num_chunks
#raw data chunk
python ./split_chunk_json.py "./submit_null.json" "bird_dev_ex_chunk" $num_chunks
#run llm
gpu_id=0
curr_batch=0
for index in $(seq 1 $num_chunks); do
  if [ $curr_batch -ge $batch_per_gpu ]; then
    gpu_id=$((gpu_id+1))
    curr_batch=1
  else
    curr_batch=$((curr_batch+1))

  fi
  echo "Run chunk : $index, Num sample: `python ./get_json_len.py ${prefix}_${index}.json`"
  # gpu_id=$(awk "BEGIN {print $index/$num_gpus}")
  # gpu_id=$(printf "%.0f" $gpu_id)
  echo "Gpu ID: $gpu_id"
  CUDA_VISIBLE_DEVICES=$((index-1)) python ./infer_bird.py --index $index --llm_path $1 --sic_path ./submit_sic_ckp/ --table_num 6 --column_num 10 --dataset_path "./${prefix}_${index}.json" --max_tokens 4096 --max_new_tokens 256 --chunk_len $chunk_len --model_type $model_type &
done
wait
#merge the json files
python ./merge_predict_json.py "predict_dev_" "./predict_dev.json"
#evaluate
bash ./debug_evaluation/run_evaluation.sh ./predict_dev.json #evaluate
#end time
end_time=$(date +%s)
echo "Runtime: $(($end_time - $start_time)) seconds"
