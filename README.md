Official Codebase for [*Active Preference Optimization for Sample Efficient RLHF*](https://arxiv.org/abs/2402.10500) by Nirjhar Das, Souradip Chakrabory, Aldo Pacchiano and Sayak Ray Chowdhury.

## Installation
The code base has dependency on basic packages listed in [requirements.txt](./requirements.txt). It requires a conda installation and the environment can be set up via the following command:
```
$ conda create -n trl
$ conda activate trl
$ conda install -r requirements.txt
```

## Usage
After the environment is set, the dataset used for the [Anthropic-HH](https://huggingface.co/datasets/llm-blender/Unified-Feedback/viewer/hh-rlhf) experiments in the paper can be generated using the [data_preparation.ipynb](./data_preparation.ipynb) Jupyter notebook. After dataset creation, the results can be replicated using the bash script [```runall.sh```](./runall.sh). Alternatively, the Jupyter notebook [apo_notebook.ipynb](./apo_notebook.ipynb) can be used. The plots can be generated via [plots.ipynb](./plots.ipynb).

## References
If you find this work useful, please consider citing:
~~~bibtex
@article{das2024active,
    title={Active Preference Optimization for Sample Efficient RLHF},
    author={Das, Nirjhar and Chakraborty, Souradip and Pacchiano, Aldo and Chowdhury, Sayak Ray},
    year={2024},
    eprint={2402.10500},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~