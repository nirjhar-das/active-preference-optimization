 #!/bin/sh

conda activate trl
python scripts/reward_learning.py -d Anthropic -m Gemma -o ./output -e 100 -s 20 -i apo -z True -f le
python scripts/reward_learning.py -d Anthropic -m Gemma -o ./output -e 100 -s 20 -i random -z True -f le
python scripts/reward_evaluation.py -d Anthropic -m Gemma -o ./output -i apo -z True -f le
python scripts/reward_evaluation.py -d Anthropic -m Gemma -o ./output -i random -z True -f le
python scripts/ppo_model_training.py -d Anthropic -m Gemma -o ./output -i apo -z True -f le -n 32
python scripts/ppo_model_training.py -d Anthropic -m Gemma -o ./output -i random -z True -f le -n 32
python scripts/model_evaluation.py -d Anthropic -m Gemma -o ./output -z True -n 2000 -i apo -f le
python scripts/model_evaluation.py -d Anthropic -m Gemma -o ./output -z True -n 2000 -i random -f le
conda deactivate