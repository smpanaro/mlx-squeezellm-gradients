from typing import Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import mlx.core as mx
import numpy as np
import torch

import kmeans1d

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils import cached_file, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from safetensors import safe_open
from safetensors.mlx import save_file

from jsonargparse import CLI
from tqdm import tqdm

CENTROID_SUFFIX = "centroids"
INPUT_SCALE_SUFFIX = "input_scale"
OUTPUT_SCALE_SUFFIX = "output_scale"
QUERY_SUFFIX = "query"
KEY_SUFFIX = "key"
VALUE_SUFFIX = "value"

"""
Not directly from the SqueezeLLM paper, but a useful addition is to
pre-compute per-tensor centroids after applying a scaling factor.
"""

@dataclass
class WeightMetadatum:
    name: str
    class_name: str
    base_model_prefix: str

def cluster_dict(d: dict[str, any]) -> mx.array:
    return cluster(d["name"], d["weight"], d["sensitivities"], d["num_clusters"])

def cluster(name: str, weight: mx.array, sensitivities: mx.array, num_clusters: int):
    weight_np = np.array(mx.flatten(weight.T), copy=False)
    sensitivities_np = np.array(mx.flatten(sensitivities.T), copy=False)

    unique_weight, unique_inverse = np.unique(weight_np, return_inverse=True)
    unique_sensitivities = np.bincount(unique_inverse, weights=sensitivities_np)

    eps = 0
    while True:
        if eps != 0:
            print(f"Encountered numerical instability in {name}. Trying with eps: {eps}")

        kmeans = kmeans1d.cluster(unique_weight, num_clusters, weights=unique_sensitivities + eps)
        if not np.isnan(kmeans.centroids).any():
            break
        elif eps == 0:
            eps = 1e-12
        elif eps > 1e-4:
            print(f"WARNING: KMeans significantly interrupted by numerical instability for {name}.")
        else:
            eps *= 100

    return mx.array(kmeans.centroids, dtype=mx.float32)

# TODO: Consider switching to this.
def column_clusters(weight: mx.array, sensitivities: mx.array, num_clusters: int):
    # Empirically, the relative distribution is usually the same but sometimes flipped across the y-axis.
    # It would be nice if there was a way to do this without clustering, but all the heuristics
    # I've tried so far are lacking (ie result in worse perplexity).
    def compute_column_centroids(inputs):
        col, sense, num_clusters = inputs
        kmeans = kmeans1d.cluster(col, num_clusters, weights=sense)
        return kmeans.centroids

    # Do this in two steps since it is a bottleneck.
    # First compute all the per-column clusters.
    tasks = []
    for col, sense in zip(weight.T, sensitivities.T):
        tasks.append((col, sense, num_clusters))

    centroids = mx.zeros((len(tasks), num_clusters))
    with ThreadPoolExecutor() as executor:
        for i, result in enumerate(executor.map(compute_column_centroids, tasks)):
            centroids[i] = result

    # Then compute the scales based on them.
    @mx.compile
    def compute_all_scales(centroids: mx.array, stds: mx.array):
        scales = mx.zeros_like(stds)
        for i, (cs, std) in enumerate(zip(centroids, stds)):
            flip = mx.where(cs.min().abs() > cs.max(), -1, 1)
            scale = flip * std
            scales[i] = scale
        return scales

    stds = weight.var(axis=0).sqrt()
    scales = compute_all_scales(centroids, stds)
    return scales

# @mx.compile # bad! wrong results
def scale_weight(weight: mx.array, class_name: str) -> Tuple[mx.array, mx.array]:
    axis = 0 if class_name in ["Conv1D"] else 1
    # ddof = 1 to match PyTorch.
    std = weight.var(axis=axis, keepdims=True, ddof=1).sqrt()
    return mx.divide(weight, std), std

def get_key_func(hf_keys, other_keys, base_model_name: str, other_keys_name: str):
    """Return a func that maps other_keys to hf_keys."""

    # TODO: Handle when other_keys are HF keys.

    # Perform a two-step mapping from HF keys <> generic keys <> other keys.

    # Generic keys are defined here, and map to values based on the
    # base_model_name, which comes from HF.
    common_to_hf = {
        "layer": None, # ie layer, layers, h
        "attn.qkv": None,
        "attn.q_proj": None,
        "attn.k_proj": None,
        "attn.v_proj": None,
        "attn.o_proj": None,
        "mlp.up_proj": None,
        "mlp.down_proj": None,
        "mlp.gate_proj": None,
    }

    if base_model_name == "gpt_neox":
        common_to_hf["layer"] = "layers"
        common_to_hf["attn.qkv"] = "attention.query_key_value"
        common_to_hf["attn.q_proj"] = None
        common_to_hf["attn.k_proj"] = None
        common_to_hf["attn.v_proj"] = None
        common_to_hf["attn.o_proj"] = "attention.dense"
        common_to_hf["mlp.up_proj"] = "mlp.dense_h_to_4h"
        common_to_hf["mlp.down_proj"] = "mlp.dense_4h_to_h"
        common_to_hf["mlp.gate_proj"] = None
    if base_model_name == "model": # llama, probably not the best way to detect this
        common_to_hf["layer"] = "layers"
        common_to_hf["attn.qkv"] = None
        common_to_hf["attn.q_proj"] = "self_attn.q_proj"
        common_to_hf["attn.k_proj"] = "self_attn.k_proj"
        common_to_hf["attn.v_proj"] = "self_attn.v_proj"
        common_to_hf["attn.o_proj"] = "self_attn.o_proj"
        common_to_hf["mlp.up_proj"] = "mlp.up_proj"
        common_to_hf["mlp.down_proj"] = "mlp.down_proj"
        common_to_hf["mlp.gate_proj"] = "mlp.gate_proj"
    else:
        raise ValueError(f"Unknown base_model_name: {base_model_name}")

    print(other_keys)

    # Other keys can come from one of several transformer implementations.
    # Map from implementation-specific keys to HF keys by way of generic keys.
    longest_other_key = max(other_keys, key=lambda k: len([c for c in k if c == "."]))
    from_other_to_hf = {}
    if longest_other_key.startswith("model."):
        # mlx-lm
        print(f"Detected mlx-lm {other_keys_name}.")
        from_other_to_hf["model"] = ""
        from_other_to_hf["layers"] = common_to_hf["layer"]
        # from_other_to_hf[not supported] = common_to_hf["attn.qkv"]
        from_other_to_hf["self_attn.q_proj"] = common_to_hf["attn.q_proj"]
        from_other_to_hf["self_attn.k_proj"] = common_to_hf["attn.k_proj"]
        from_other_to_hf["self_attn.v_proj"] = common_to_hf["attn.v_proj"]
        from_other_to_hf["self_attn.o_proj"] = common_to_hf["attn.o_proj"]
        from_other_to_hf["mlp.up_proj"] = common_to_hf["mlp.up_proj"]
        from_other_to_hf["mlp.down_proj"] = common_to_hf["mlp.down_proj"]
        from_other_to_hf["mlp.gate_proj"] = common_to_hf["mlp.gate_proj"]
    elif longest_other_key.startswith("transformer."):
        # lit-gpt
        print(f"Detected lit-gpt {other_keys_name}.")
        from_other_to_hf["transformer"] = "" # base model name
        from_other_to_hf["h"] = common_to_hf["layer"]
        from_other_to_hf["attn.attn"] = common_to_hf["attn.qkv"] # from official repo
        from_other_to_hf["attn.q_proj"] = common_to_hf["attn.q_proj"] # from my fork
        from_other_to_hf["attn.k_proj"] = common_to_hf["attn.k_proj"] # from my fork
        from_other_to_hf["attn.v_proj"] = common_to_hf["attn.v_proj"] # from my fork
        from_other_to_hf["attn.proj"] = common_to_hf["attn.o_proj"]
        # These vary by model.
        if base_model_name == "gpt_neox":
            from_other_to_hf["mlp.fc"] = common_to_hf["mlp.up_proj"]
            from_other_to_hf["mlp.proj"] = common_to_hf["mlp.down_proj"]
        else:
            raise ValueError(f"Unknown base_model_name for {other_keys_name}: {base_model_name}")
    else:
        raise ValueError(f"Unknown key format for {other_keys_name}: {longest_other_key}")

    def map_key(key):
        for source, dest in from_other_to_hf.items():
            if dest is None:
                continue
            if dest == "":
                key = key.replace(f"{source}.", f"{dest}")
            else:
                key = key.replace(f"{source}.", f"{dest}.")
        return key

    return map_key

def get_inverse_key_func(key_func, all_other_keys):
    inverse_map = {key_func(k): k for k in all_other_keys}
    def map_inverse_key(k):
        return inverse_map.get(k, k)
    return map_inverse_key

def generate_centroids(
        weights: safe_open,
        sensitivities: safe_open,
        nbits: int,
        weight_metadata: dict[str, WeightMetadatum],
) -> dict[str, mx.array]:
    num_clusters = 2**nbits

    result = {}

    base_model_name = weight_metadata[list(weight_metadata.keys())[0]].base_model_prefix

    # metadata keys are HF keys but without the base_model prefix
    # weights are HF keys, sometimes with and sometimes without the base_model prefix
    # sensitivities are a mixed bag, some come from HF, some from lit-gpt, some from mlx-lm
    sense_to_hf_key = get_key_func(weight_metadata.keys(), sensitivities.keys(), base_model_name, "sensitivities")
    hf_to_sense_key = get_inverse_key_func(sense_to_hf_key, sensitivities.keys())

    weight_to_hf_key = lambda k: k.replace(f"{base_model_name}.", "")
    hf_to_weight_key = get_inverse_key_func(weight_to_hf_key, weights.keys())

    # for sense_key in sensitivities.keys():
    #     print(sense_key, "->", sense_to_hf_key(sense_key))
    # print("--")

    weight_keys = weights.keys()
    weight_keys = [weight_to_hf_key(k) for k in weight_keys] # Some models have this some don't.
    weight_keys = [k for k in weight_keys if "weight" in k]
    weight_keys = [k for k in weight_keys if "lm_head" not in k]
    weight_keys = [k for k in weight_keys if "embed_in" not in k]
    weight_keys = [k for k in weight_keys if "embed_out" not in k]
    allowed_class_names = ["Linear", "Conv1D", "Conv2D"]
    weight_keys = [k for k in weight_keys
                   if weight_metadata[k].class_name in allowed_class_names]

    # Sync.
    # for key in tqdm(weight_keys):
    #     if "h.0.attn.c_proj" not in key:
    #         continue
    #     weight = weights.get_tensor(key)
    #     md = weight_metadata[key]
    #     weight, scales = scale_weight(weight, md.class_name)
    #     sensitivity = sensitivities.get_tensor(f"{md.base_model_prefix}.{key}")
    #     clusters = cluster(key, weight, sensitivity, num_clusters)

    #     print(key)
    #     if "h.0.attn.c_proj" in key:
    #         print(weights.get_tensor(key)[:5,:5])
    #         print(scales[:10])
    #         print(clusters)

    #         import sys; sys.exit(0)

    #     result[f"{md.base_model_prefix}.{key}.{CENTROID_SUFFIX}"] = clusters
    #     result[f"{md.base_model_prefix}.{key}.{OUTPUT_SCALE_SUFFIX}"] = scales

    # Async.
    with ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        args = {}
        future_to_key = {}
        for key in tqdm(weight_keys):
            if "query_key_value" in key and key not in sensitivities.keys():
                print(key, "not found in sensitivities. Skipping. TODO: Implement this.")
                continue

            weight = weights.get_tensor(hf_to_weight_key(key))
            md = weight_metadata[key]
            weight, scales = scale_weight(weight, md.class_name)
            sensitivity = sensitivities.get_tensor(hf_to_sense_key(key))
            args[key] = (key, weight, sensitivity, num_clusters)

            result[f"{md.base_model_prefix}.{key}.{OUTPUT_SCALE_SUFFIX}"] = scales

            future_to_key[executor.submit(cluster, *args[key])] = key

        # future_to_key = {executor.submit(cluster, *args): key for key, args in args.items()}
        for future in tqdm(as_completed(future_to_key), total=len(future_to_key)):
            key = future_to_key[future]
            try:
                centroids = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (key, exc))
            else:
                md = weight_metadata[key]
                result[f"{md.base_model_prefix}.{key}.{CENTROID_SUFFIX}"] = centroids

    return result

def get_weight_metadata(model_name: str) -> dict[str, WeightMetadatum]:
    # Load the model on the meta device (without allocating space for weights).
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
    md = {}
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight_key = f"{name}.weight".replace(model.base_model_prefix + ".", "")
        md[weight_key] = WeightMetadatum(name, module.__class__.__name__, model.base_model_prefix)
        # print(model.base_model_prefix, name, weight_key, module.__class__.__name__)
    return md

class CombinedSafeOpen:
    def __init__(self, safe_opens: list[safe_open]):
        self.safe_opens = safe_opens

    def keys(self):
        keys = []
        for e in self.safe_opens:
            keys.extend(e.keys())
        return keys

    def get_tensor(self, key):
        for e in self.safe_opens:
            if key in e.keys():
                return e.get_tensor(key)
        assert False, f"key {key} not found!"

def main(
        model_name: str = "gpt2",
        nbits: int = 4,
        sensitivities_path: str = None,
):
    # TODO: Do this dynamically.
    input_path_one = cached_file(model_name, "model-00001-of-00002.safetensors")# SAFE_WEIGHTS_NAME)
    weights_one = safe_open(input_path_one, "mlx", "cpu")

    input_path_two = cached_file(model_name, "model-00002-of-00002.safetensors")# SAFE_WEIGHTS_NAME)
    weights_two = safe_open(input_path_two, "mlx", "cpu")

    weights = CombinedSafeOpen([weights_one, weights_two])

    # input_path = cached_file(model_name, SAFE_WEIGHTS_NAME)
    # weights = safe_open(input_path, "mlx", "cpu")

    sensitivities = safe_open(sensitivities_path, "mlx", "cpu")
    weight_metadata = get_weight_metadata(model_name)

    centroids = generate_centroids(weights, sensitivities, nbits, weight_metadata)
    model_parts = model_name.split("/")
    if len(model_parts) > 1:
        model_dir = "".join(model_parts[:-1])
        os.makedirs(model_dir, exist_ok=True)

    metadata = {
        "info": "Contains centroids and input scales for each weight tensor.",\
# If the attention projection for query, key and values is combined,\
# the split versions are generated with extra suffixes.",
        "model_name": model_name,
        "nbits": nbits,
        "centroid_suffix": CENTROID_SUFFIX,
        "input_scale_suffix": INPUT_SCALE_SUFFIX,
        "output_scale_suffix": OUTPUT_SCALE_SUFFIX,
        "query_suffix": QUERY_SUFFIX,
        "key_suffix": KEY_SUFFIX,
        "value_suffix": VALUE_SUFFIX,
    }
    metadata = {k: str(v) for k, v in metadata.items()}
    save_file(centroids, f"{model_name}-{nbits}bit-output-scaled-centroids.safetensors", metadata=metadata)

if __name__ == "__main__":
    CLI(main)
