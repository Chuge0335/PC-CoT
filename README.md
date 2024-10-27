# PC-CoT

This is the official code repository for [ACL2024 paper: "Mitigating Boundary Ambiguity and Inherent Bias for Text Classification in the Era of Large Language Models".](https://arxiv.org/abs/2406.07001)

## Installation Guide

Clone this repository and install the required packages.

```bash
pip install -r requirements.txt
```

## Preparation Before Inference

Before starting the formal training, it is necessary to prepare the vectorized labels in advance for use during the self-reduce phase to reduce unnecessary time overhead, as well as to sample examples for context learning.

```bash
bash scripts/get_vector.sh
bash scripts/get_fewshot.sh
```
Set the openai-key in openai_client.py

```bash
api_keys = {'put your key here': (1, 'https://api.openai.com/v1')}
```
## Model Inference

Run the following command to execute all baseline methods and the methods we proposed in the paper, including evaluation metrics and result files.

```bash
bash scripts/run.sh
```

## Citation

If you would like other researchers to cite your work, please provide the citation in BibTeX format.

```bibtex
@article{PC-COT,
  title={Mitigating Boundary Ambiguity and Inherent Bias for Text Classification in the Era of Large Language Models},
  author={Zhenyi Lu #, Jie Tian #, Wei Wei*, Xiaoye Qu, Yu Cheng, Wenfeng Xie, Dangyang Chen},
  booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
  year={2024}
}
```
