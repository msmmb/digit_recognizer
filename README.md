Install required libraries
```sh
pip install -r requirements.txt
```

## Test model
It tests a pre-trained model (92% accuracy)
```sh
python3 test.py
```

## Train model
Since the model doesn't support batches and paralelization, it takes around 30 minutes to run a single epoch

```sh
python3 train.py
```
