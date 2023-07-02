Install required libraries

```sh
pip install -r requirements.txt
```

## Test model
It tests a pre-trained model (90% accuracy)
```
python3 test.py
```

## Train model
Since the model doesn't support batches nor paralelization, it takes around 30 minutes to run a single epoch

```py
python3 train.py
```