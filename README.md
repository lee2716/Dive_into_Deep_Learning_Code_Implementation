# Dive into Deep Learning — My Code Implementations

My hands-on notes and code while working through [Dive into Deep Learning (D2L)](https://d2l.ai/) by Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola. Each notebook re-implements a model from scratch, with my own explanations of *why* each step is done.

## 📓 Notebooks

| # | Notebook | Topic | Framework | D2L reference | Open |
|---|----------|-------|-----------|---------------|------|
| 1 | [`myfirstNN.ipynb`](myfirstNN.ipynb) | Approximating a 1-D function with a dense network; data generators, MSE loss | TensorFlow / Keras | [Ch. 5 MLPs](https://d2l.ai/chapter_multilayer-perceptrons/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/myfirstNN.ipynb) |
| 2 | [`mnistDense.ipynb`](mnistDense.ipynb) | MNIST digit classification: softmax regression vs. one hidden layer, one-hot labels, training curves | TensorFlow / Keras | [Ch. 4 Linear Classification](https://d2l.ai/chapter_linear-classification/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/mnistDense.ipynb) |
| 3 | [`my_resnet.ipynb`](my_resnet.ipynb) | ResNet-18 from scratch: residual blocks, shape tracing, training on Fashion-MNIST | PyTorch | [§8.6 ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/my_resnet.ipynb) |

## 🗺️ Roadmap

- [x] Multilayer perceptrons (Keras)
- [x] Softmax classification on MNIST (Keras)
- [x] ResNet (PyTorch)
- [ ] Linear regression from scratch (PyTorch)
- [ ] LeNet & classic CNNs (AlexNet, VGG)
- [ ] Batch normalization
- [ ] RNN / LSTM / GRU
- [ ] Attention & Transformer

## 🚀 Getting Started

The quickest way is the **Open in Colab** button in the table above — no setup needed (choose a GPU runtime for ResNet).

To run locally:

```bash
git clone https://github.com/lee2716/Dive_into_Deep_Learning_Code_Implementation.git
cd Dive_into_Deep_Learning_Code_Implementation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## 📚 References

- Zhang, Lipton, Li, Smola. *Dive into Deep Learning*. <https://d2l.ai/>
- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016. <https://arxiv.org/abs/1512.03385>

## 🤝 Contributing

This is a personal learning project, but issues, corrections and discussion are very welcome. Let's learn together! 😊

## 📄 License

[MIT](LICENSE)
