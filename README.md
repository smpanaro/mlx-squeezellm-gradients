# mlx-squeezellm-gradients

Generate SqueezeLLM-style gradients using MLX and faster than PyTorch MPS. [Original Implementation](https://github.com/kssteven418/SqueezeLLM-gradients)

## Installation
```shell
conda create -n mlx-squeezellm-grads python=3.11 -y
conda activate mlx-squeezellm-grads
conda install --file requirements.txt
# Later, when you're done.
conda deactivate
```

## Usage

```shell
python run.py --model_name meta-llama/Llama-2-7b-hf --gpu True --nsamples 100
```