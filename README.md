# llama2-easylm
### Installation
```
conda create -n easylm python=3.10
bash scripts/tpu_vm_setup.sh
```
### Inference on LLaMA-2
First, transform your test data in the same format as `example_data/test_examples.json`, and then run:
```
bash predict.sh
```
The output would be stored in the same format as ``example_data/test_output.json``
### Finetuning LLaMA-2
Prepare your training data in the same format as `example_data/train_examples.json`, and then run:
```
bash train.sh
```
You can specify your wandb api key to store and visualize the logs and training curves.
