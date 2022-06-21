# HandMeThat-Release

## Third-party modules
* Jacinle
* HACL-PyTorch
* XTX
* alfworld

## Training and evaluation

### DRRN/offlineDRRN

To train the model (e.g., 'DRRN' with 'fully' observable setting):
```
python scripts/train_rl.py --model DRRN --observability fully
```

To evaluate the model (e.g., validate) on specific hardness level (e.g., level1):
```
python scripts/eval_rl.py --model DRRN --observability fully --level level1 --eval_split validate --memory_file memory_1 --weight_file weights_1
```

### Seq2Seq

To train the model (e.g., 'partially' observable setting):
```
python scripts/train_seq.py --observability partially
```

To evaluate the model (e.g., test) on specific hardness level (e.g., level1):
```
python scripts/eval_seq.py --observability partially --level level1 --eval_split test --eval_model_name weights_5000.pt
```

### Random Agent

To test the random agent:

```
python scripts/eval.py --agent random --level level1 --eval_split test
```

### Play with the environment
To play with the environment on your own:
```
python scripts/eval.py --agent human --level level1 --eval_split test
```
