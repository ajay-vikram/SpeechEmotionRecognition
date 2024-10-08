# SpeechEmotionRecognition
Speech Emotion Classification on 4 datasets using Legendre Memory Units (LMU)

## File Descriptions
- *SER.ipynb*: generate train and test tensors
- *utils.py*: modify training configurations

## Datasets
Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio </br>
Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D): https://www.kaggle.com/datasets/ejlok1/cremad </br>
Surrey Audio-Visual Expressed Emotion (SAVEE): https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee </br>
Toronto emotional speech set (TESS): https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

## Training

```
python main.py --train
```

## Training Results

| **Model**                        | **Parameters** | **Training Accuracy** | **Validation Accuracy** |
|-----------------------------------|----------------|-----------------------|-------------------------|
| **CNN1**                          | 0.3M           | 59%                   | 60%                     |
| **CNN2**                          | 7.2M           | 99%                   | 94%                     |
| **LSTM**                          | 51K            | 69%                   | 62%                     |
| **CNN_LSTM**                      | 3.8M           | 87%                   | 78%                     |
| **LMU(128, 64, 32, 16) d=1024**   | **1.3M**           | **97%**                   | **92%**                     |

## LMU Architecture

The LMU is mathematically derived to orthogonalize its continuous-time history – doing
so by solving *d* coupled ordinary differential equations (ODEs), whose phase space
linearly maps onto sliding windows of time via the Legendre polynomials up to degree
*d* − 1 (the example for *d* = 12 is shown below).

![](https://i.imgur.com/Uvl6tj5.png)


A single LMU cell expresses the following computational graph, which takes in an input
signal, **x**, and couples a optimal linear memory, **m**, with a nonlinear hidden
state, **h**. By default, this coupling is trained via backpropagation, while the
dynamics of the memory remain fixed.

![](https://i.imgur.com/IJGUVg6.png)


The discretized **A** and **B** matrices are initialized according to the LMU's
mathematical derivation with respect to some chosen window length, **θ**.
Backpropagation can be used to learn this time-scale, or fine-tune **A** and **B**,
if necessary.

Both the kernels, **W**, and the encoders, **e**, are learned. Intuitively, the kernels
learn to compute nonlinear functions across the memory, while the encoders learn to
project the relevant information into the memory (see `paper
<https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks.pdf>`_ for details).

## References
- https://github.com/hrshtv/pytorch-lmu
