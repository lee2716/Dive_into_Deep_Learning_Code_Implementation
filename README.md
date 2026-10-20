# Dive into Deep Learning — My Code Implementations

My hands-on notes and code while working through [Dive into Deep Learning (D2L)](https://d2l.ai/) by Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola. Each notebook re-implements a model from scratch, with my own explanations of *why* each step is done.

## 📓 Notebooks

| # | Notebook | Topic | Framework | D2L reference | Open |
|---|----------|-------|-----------|---------------|------|
| 1 | [`linear_regression.ipynb`](linear_regression.ipynb) | Linear regression from scratch: data iterator, squared loss, minibatch SGD; concise `nn.Linear` version; closed-form check | PyTorch | [Ch. 3 Linear Regression](https://d2l.ai/chapter_linear-regression/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/linear_regression.ipynb) |
| 2 | [`myfirstNN.ipynb`](myfirstNN.ipynb) | Approximating a 1-D function with a dense network; data generators, MSE loss | TensorFlow / Keras | [Ch. 5 MLPs](https://d2l.ai/chapter_multilayer-perceptrons/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/myfirstNN.ipynb) |
| 3 | [`mnistDense.ipynb`](mnistDense.ipynb) | MNIST digit classification: softmax regression vs. one hidden layer, one-hot labels, training curves | TensorFlow / Keras | [Ch. 4 Linear Classification](https://d2l.ai/chapter_linear-classification/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/mnistDense.ipynb) |
| 4 | [`classic_cnns.ipynb`](classic_cnns.ipynb) | LeNet trained on Fashion-MNIST; AlexNet and VGG-11 architectures with layer-by-layer shape tracing | PyTorch | [Ch. 7 CNNs](https://d2l.ai/chapter_convolutional-neural-networks/index.html), [§8.1–8.2](https://d2l.ai/chapter_convolutional-modern/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/classic_cnns.ipynb) |
| 5 | [`batch_norm.ipynb`](batch_norm.ipynb) | Batch normalization from scratch (FC + conv, train vs. predict mode); LeNet with vs. without BN | PyTorch | [§8.5 Batch Norm](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/batch_norm.ipynb) |
| 6 | [`my_resnet.ipynb`](my_resnet.ipynb) | ResNet-18 from scratch: residual blocks, shape tracing, training on Fashion-MNIST | PyTorch | [§8.6 ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/my_resnet.ipynb) |
| 7 | [`rnn_lstm_gru.ipynb`](rnn_lstm_gru.ipynb) | Character-level language models on *The Time Machine*: RNN and LSTM from scratch, gradient clipping, perplexity, `nn.RNN` / `nn.GRU` / `nn.LSTM` compared | PyTorch | [Ch. 9 RNNs](https://d2l.ai/chapter_recurrent-neural-networks/index.html), [Ch. 10](https://d2l.ai/chapter_recurrent-modern/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/rnn_lstm_gru.ipynb) |
| 8 | [`transformer.ipynb`](transformer.ipynb) | Transformer from scratch: scaled dot-product & multi-head attention, positional encoding, masks; English→French translation with BLEU and attention heatmaps | PyTorch | [Ch. 11 Attention & Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lee2716/Dive_into_Deep_Learning_Code_Implementation/blob/main/transformer.ipynb) |

## 🗺️ Roadmap

- [x] Linear regression (Ch. 3)
- [x] Softmax classification & MLPs (Ch. 4–5)
- [x] CNNs: LeNet, AlexNet, VGG (Ch. 7–8)
- [x] Batch normalization (§8.5)
- [x] ResNet (§8.6)
- [x] RNN / GRU / LSTM (Ch. 9–10)
- [x] Attention & Transformer (Ch. 11)

Possible next steps: optimization algorithms (Ch. 12), computer vision fine-tuning (Ch. 14), pretrained BERT (Ch. 15).

The notebooks are saved with their outputs (training logs, plots) from real runs on a CPU. Heavier training (AlexNet, VGG, ResNet-18) needs a GPU — open those notebooks in Colab with a GPU runtime.

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
- LeCun et al. *Gradient-Based Learning Applied to Document Recognition* (LeNet). 1998.
- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS 2012.
- Simonyan, Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG). <https://arxiv.org/abs/1409.1556>
- Ioffe, Szegedy. *Batch Normalization*. ICML 2015. <https://arxiv.org/abs/1502.03167>
- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016. <https://arxiv.org/abs/1512.03385>
- Hochreiter, Schmidhuber. *Long Short-Term Memory*. Neural Computation, 1997.
- Cho et al. *Learning Phrase Representations using RNN Encoder–Decoder* (GRU). <https://arxiv.org/abs/1406.1078>
- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. <https://arxiv.org/abs/1706.03762>

## 🤝 Contributing

This is a personal learning project, but issues, corrections and discussion are very welcome. Let's learn together! 😊

## 📄 License

[MIT](LICENSE)
