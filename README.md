# Simple Predictive Coding

This repository contains a relatively simple implementation of [predictive coding](https://en.wikipedia.org/wiki/Predictive_coding), a learning algorithm that, unlike [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), is biologically plausible.
You should not use this repository to train large models: use [PCX](https://github.com/liukidar/pcx), a JAX-based predictive coding framework, instead.
You should use this repository if you want to learn about how predictive coding works.

`main.py` contains a heavily commented predictive coding implementation.
The core implementation is in lines 49-125, but most of this is comments: there are only 39 lines of code.
Feel free to read it, run some experiments, and write an issue or send me an email if you think anything is unclear!

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train a 3-layer MLP on MNIST:

```bash
python main.py --model mlp --dataset mnist
```

Train a VGG-5 model on CIFAR10:

```bash
python main.py --model vgg5 --dataset cifar10
```

## License

Simple Predictive Coding is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
