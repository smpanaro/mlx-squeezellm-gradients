import os
from functools import partial

from tqdm import tqdm
from datasets import load_dataset
from jsonargparse import CLI

from mlx_lm import load

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Yields nsamples of (input, target) pairs from the C4 dataset.
    """
    import random
    random.seed(seed)

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', token=False
    )

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='np')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        input_ids_len = trainenc.input_ids.shape[1]
        i = random.randint(0, input_ids_len - seqlen - 1) if input_ids_len != seqlen else 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.copy()
        tar[:, :-1] = -100
        yield (inp, tar)

def collect_sensitivities(model_name_or_path, model, output_path, nsamples, seqlen, device, tokenizer):
    # Baesd on SqueezeLLM-gradients.

    def loss_fn(model, X):
        logits, _ = model(X)
        logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        labels = X[..., 1:].reshape(-1)
        assert logits.shape[0] == labels.shape[0], (logits.shape, labels.shape)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    seed = 42
    dataloader = get_c4(nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)

    # Match the precision of the Torch implementation.
    model.apply(lambda x: x.astype(mx.float32))

    grads = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            grads[name + ".weight"] = mx.zeros_like(mod.weight)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @mx.compile
    def step(x, grads):
        _, curr_grads = loss_and_grad_fn(model, x)
        for name, grad in tree_flatten(curr_grads):
            if name in grads:
                grads[name] += grad.square()
        return grads

    for data in tqdm(dataloader, total=nsamples):
        inp = mx.array(data[0])

        # Non-compiled version.
        # _, curr_grads = loss_and_grad_fn(model, inp)
        # for name, grad in tree_flatten(curr_grads):
        #     if name in grads:
        #         grads[name] += grad ** 2
        # mx.eval(grads)

        # Compiled version. ~10% faster.
        grads = step(inp, grads)
        mx.eval(grads)

    print(f"saving model gradient at {output_path}")
    metadata = {
        'nsamples': nsamples,
        'seed': seed,
        'seqlen': seqlen,
    }
    metadata = {k: str(v) for k,v in metadata.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mx.save_safetensors(output_path, grads, metadata)

def main(
    model_name: str = "mlx-community/Qwen1.5-0.5B-Chat",
    nsamples: int = 100,
    seqlen: int = 512,
    gpu: bool = False
):
    if "phi-2" in model_name:
        print("This model may not work since it doesn't have safetensors.")

    if not gpu:
        mx.set_default_device(mx.cpu)

    model, tokenizer = load(model_name)

    collect_sensitivities(
        model_name,
        model=model,
        output_path=f"{model_name}-grads.safetensors",
        nsamples=nsamples,
        seqlen=seqlen,
        device=None,
        tokenizer=tokenizer)

if __name__ == "__main__":
    CLI(main)