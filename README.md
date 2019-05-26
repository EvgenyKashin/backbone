# Backbone for PyTorch training loop
Will try to keep it minimalistic.

`pip install back`
```python
from back import Bone
```

## Features
- Progress bar
- Checkpoints saving/loading
- Schedulers
- Tensorboard logging
- Early stopping
- Multiple GPUs

## TODO
- [ ] Learning Rate Finder
- [ ] Notebook example
- [ ] Docs/tests

## Docker
`docker build -t backbone_pytorch .`

or

`sudo docker pull digitman/backbone_pytorch:latest`

## Examples
### SRM 
https://github.com/EvgenyKashin/SRMnet