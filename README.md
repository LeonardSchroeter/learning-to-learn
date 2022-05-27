# Machine learning of gradient-based optimization methods

This repository holds the code for the experiments in my bachelor thesis with the above title. It contains an implementation of the learning to learn approach suggested in [this paper](https://proceedings.neurips.cc/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf). It also contains all the tests that I ran for my bachelor thesis.

The tests were run with python version 3.9.5.
The dependencies can be installed with:

```
pip install -r requirements.txt
```

The experiments are listed in the experiments directory and are called in main.py.
The class `LearningToLearn` in src/train.py contains the core learning to learn algorithm.
