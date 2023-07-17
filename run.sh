CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b
CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-sft
CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-sft-v2
CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b-instruction-ppo

CUDA_VISIBLE_DEVICES=1 poetry run accelerate launch src/train.py --model_name cyberagent/open-calm-1b
CUDA_VISIBLE_DEVICES=1 poetry run accelerate launch src/train.py --model_name cyberagent/open-calm-3b
CUDA_VISIBLE_DEVICES=1 poetry run accelerate launch src/train.py --model_name cyberagent/open-calm-7b

CUDA_VISIBLE_DEVICES=0 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 0
CUDA_VISIBLE_DEVICES=1 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 1
CUDA_VISIBLE_DEVICES=2 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 2
CUDA_VISIBLE_DEVICES=3 poetry run accelerate launch src/train.py --model_name rinna/japanese-gpt-neox-3.6b --template_type 3
