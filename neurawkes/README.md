# The Neural Hawkes Process
Source code for [The Neural Hawkes Process (NeurIPS 2017)](https://arxiv.org/abs/1612.09328) runnable on GPU and CPU.

Please check out [The Neural Hawkes Particle Smoothing](https://github.com/HMEIatJHU/neural-hawkes-particle-smoothing) for PyTorch Implementation (and more). 

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

[The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328)  
[Hongyuan Mei](http://www.cs.jhu.edu/~hmei/) and [Jason Eisner](http://www.cs.jhu.edu/~jason/)
```
@inproceedings{mei2017neuralhawkes,
  author =      {Hongyuan Mei and Jason Eisner},
  title =       {The Neural {H}awkes Process: {A} Neurally Self-Modulating Multivariate Point Process},
  booktitle =   {Advances in Neural Information Processing Systems},
  year =        {2017},
  month =       dec,
  address =     {Long Beach},
  url =         {https://arxiv.org/abs/1612.09328}
}
```

## Instructions
Here are the instructions to use the code base

### Dependencies
This code is written in Python.

- **Python**: updated for **Python 3.13+** syntax and best practices.
- **Theano**: computational graphs are built on Theano.

Important note on **Python 3.13**: upstream Theano is unmaintained, and Theano (or forks like `theano-pymc`) may not provide wheels compatible with Python 3.13 on all platforms. This repo keeps the original Theano-based logic; you may need to use a compatible environment (e.g. older Python) to actually run training/testing today.

### Prepare Data
Download datasets to the 'data' folder

### Train Models
To train the model, try the command line below for detailed guide:
```
python train_models.py --help
```

### Test Models
To evaluate (dev or test) and save results, use the command line below for detailed guide:
```
python test_models_and_save.py --help
```

### Generate Sequences
To generate sequences (with trained or randomly initialized models), try the command line:
```
python generate_sequences.py --help
```

### Significant Tests
To test statistical significance by boostrapping over dev/test set, try the command line:
```
python generate_sequences.py --help
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

