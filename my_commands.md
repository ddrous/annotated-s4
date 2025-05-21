## MNIST

```python
nohup python -m s4.train dataset=mnist layer=s4 train.epochs=100 train.bsz=128 model.d_model=340 model.layer.N=64 train.sample=300 > mnist.log &
```
```
nohup python -m s4.train dataset=mnist layer=s4 train.epochs=1 train.bsz=50 train.lr=5e-3 train.lr_schedule=true model.d_model=512 model.n_layers=6 model.dropout=0.0 train.weight_decay=0.05 model.prenorm=true model.embedding=true train.sample=300 > mnist.log &
```
