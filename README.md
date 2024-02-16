# DoRA

Implementation of "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al, 2024) https://arxiv.org/pdf/2402.09353.pdf

## Demo

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

```bash
git clone https://github.com/catid/dora.git
cd dora

conda create -n dora python=3.10 -y && conda activate dora

pip install -U -r requirements.txt

python dora.py
```

## Output

```bash
Total Parameters: 11
Trainable Parameters: 11
Final Evaluation Loss: 0.1101187914609909
Total Parameters: 65
Trainable Parameters: 54
Continuing training with DoRA layers...
Final (DoRA) Evaluation Loss: 0.02946443296968937
```
