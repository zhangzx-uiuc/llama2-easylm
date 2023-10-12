#! /bin/bash

# This is the example script to serve a 7B LLaMA model on a GPU machine or
# single TPU v3-8 VM. The server will be listening on port 35009.


python -m EasyLM.models.llama.llama_predict \
    --load_llama_config='7b' \
    --load_checkpoint="params::/path/to/llama2/checkpoint" \
    --tokenizer.vocab_file='/path/to/llama2/tokenizer' \
    --mesh_dim='1,1,-1' \
    --dtype='bf16' \
    --input_length=512 \
    --seq_length=1024 \
    --prediction_input_file="./example_data/test_examples.json" \
    --prediction_output_file="./example_data/test_output.json" \
    --prediction_output_field="output"



