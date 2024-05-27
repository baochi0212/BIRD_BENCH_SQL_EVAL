import json
import argparse
import math

def split_json_file(input_file, output_prefix, num_chunks):
    with open(input_file, 'r') as file:
        data = json.load(file)
    print("Sample: ", data[0])
    total_items = len(data)
    chunk_size = math.ceil(total_items / num_chunks)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, total_items)
        chunk_data = data[start_index:end_index]

        output_file = f"{output_prefix}_{i+1}.json"
        with open(output_file, 'w') as file:
            json.dump(chunk_data, file, indent=2)

        print(f"Created chunk file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split JSON file into chunks')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_prefix', help='Prefix for the output chunk files')
    parser.add_argument('num_chunks', type=int, help='Number of chunks to split the JSON file into')

    args = parser.parse_args()

    split_json_file(args.input_file, args.output_prefix, args.num_chunks)

if __name__ == '__main__':
    main()
