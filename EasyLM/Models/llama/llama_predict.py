import dataclasses
import pprint
from functools import partial
import re
import os
from threading import Lock
from tqdm import tqdm


from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax
from transformers import GenerationConfig, FlaxLogitsProcessorList

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, get_jax_mesh, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
from EasyLM.jax_utils import with_sharding_constraint

import json


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,1,-1',
    dtype='bf16',
    input_length=64,
    seq_length=64,
    top_k=5,
    temperature=1.0,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    loglikelihood_add_bos_token=True,
    load_llama_config='',
    load_checkpoint='',
    prediction_input_file='',
    prediction_output_file='',
    prediction_output_field='',
    prediction_batch_size=1,
    template_index=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    lm_server=LMServer.get_default_config(),
)

def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()
    set_random_seed(FLAGS.seed)

    print("loading inputs")
    input_file = FLAGS.prediction_input_file
    input_lines = []
    if not os.path.exists(input_file):
        raise ValueError(f'Input file {input_file} does not exist')
    with open(input_file, 'r') as f:
        for line in f.readlines():
            input_lines.append(json.loads(line))

    input_text = [line['input'] for line in input_lines]
    input_chunks = [input_text[i:i + FLAGS.prediction_batch_size] for i in range(0, len(input_text), FLAGS.prediction_batch_size)]

    print("loading model")
    prefix_tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='left', padding_side='left'
    )
    tokenizer = LLaMAConfig.get_tokenizer(
        FLAGS.tokenizer, truncation_side='right', padding_side='right'
    )

    with jax.default_device(jax.devices("cpu")[0]):
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

        hf_model = FlaxLLaMAForCausalLM(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False
        )
        # params = jax.device_put(params, device=jax.devices("cpu")[0])

    model_ps = match_partition_rules(
        LLaMAConfig.get_partition_rules(), params
    )
    print(model_ps)
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.dtype)
    )

    @partial(
        pjit,
        in_axis_resources=(model_ps, PS(), PS(), PS()),
        out_axis_resources=(PS(), PS())
    )
    def forward_generate(params, rng, batch, temperature):
        batch = with_sharding_constraint(batch, PS('dp'))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            logits_processor=FlaxLogitsProcessorList(
                [FlaxTemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=FLAGS.do_sample,
                num_beams=FLAGS.num_beams,
                top_k=FLAGS.top_k,
                top_p=FLAGS.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    print(mesh)
    assert len(mesh.shape) == 3, 'MP mesh must be 2D'
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()

    def generate(text, temperature):
        print(text)
        nonlocal sharded_rng
        inputs = prefix_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=FLAGS.input_length,
            return_tensors='np',
        )
        print(inputs)
        batch = dict(
            input_tokens=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        with mesh:
            output, sharded_rng = forward_generate(
                params, sharded_rng, batch, temperature
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(tokenizer.batch_decode(output)):
            if tokenizer.eos_token in text:
                text = text.split(tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)

        return output_text
    
    outputs = [generate(text_chunk, FLAGS.temperature) for text_chunk in tqdm(input_chunks)]
    outputs = [item for sublist in outputs for item in sublist]

    with open(FLAGS.prediction_output_file, 'w') as f:
        for line, output in zip(input_lines, outputs):
            line[FLAGS.prediction_output_field] = output
            f.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    mlxu.run(main)
