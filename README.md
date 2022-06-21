# HandMeThat-Release

## Third-party modules
* Jacinle
* HACL-PyTorch
* XTX
* alfworld

## Training and evaluation

### DRRN/offlineDRRN

To train the model:
```
python scripts/train_rl.py --model [DRRN/offlineDRRN] --observability [fully/partially]
```

To evaluate the model (validate/test) on specific hardness level (e.g., level1):
```
python scripts/eval_rl.py --model [DRRN/offlineDRRN] --observability [fully/partially] --level level1 --eval_split test --memory_file memory_1 --weight_file weights_1
```
