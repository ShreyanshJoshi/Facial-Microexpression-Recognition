# Facial Microexpression Recognition using Spatio-Temporal CNNs

## Installation 
This repo contains the Keras (Tensorflow) implementation of our work. The code with recent python versions, e.g 3.9 and recent tensorflow versions, e.g 2.6.0. Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda install tensorflow=2.6
conda install -c conda-forge matplotlib
conda install pandas
conda install keras=2.6.0
conda install -c anaconda scikit-learn keras
```

The following need to be adapted in order to run the code on your own machine:
- Change the file path (*dir*) to the SAMM dataset in `utils/augmentation.py`, `utils/common.py`, `utils/miscellaneous.py`

The SAMM dataset should be downloaded separately and saved to the path described in the above files by the variable *dir*.
## Introduction
Micro-expression is a spontaneous expression that occurs when a person tries to mask their inner emotion, and can neither be forged nor suppressed. Micro-expressions are low-intensity, short-duration facial expressions that are likely signs of concealed emotions and may occur consciously or subconsciously. Typically, a microexpression lasts less than 0.5 seconds and consequently, facial microexpression detection is an incredibly difficult task (for humans and machines alike) and, even more so when it comes to categorizing them into different categories. Notwithstanding the few attempts made for recognizing (classifying) micro-expressions, the problem is far from being a solved problem, which is depicted by the low rate of accuracy and/or high computational complexity shown by the state-of-the-art methods. Addressing these issues, this study proposes architectures outperforming most of the existing methods coupled with a lower computational cost. Experiments are conducted on a publicly available dataset: SAMM (Spontaneous Actions and Micro-Movements). We perform both binary and multiclass classification (across 7 classes) for classification of facial microexpressions. For each of them, 3 architectures have been used, namely, **3DCNN**, **CNN+LSTM (Time Distributed)** and **CNN-LSTM (ConvLSTM2D)**. These models are capable of extracting the temporal correlation between the frames of a video. Apart from this, a **majority voting policy was also implemented using 2D CNNs**, to see the effect of not extracting the temporal correlations and relying only on spatial features. Finally, to check the generalizability of our models, we tested them for binary classification on CASME II dataset, achieving a best accuracy of 66%.

## Organization
- *main.py* and *main_2dcnn_with_majority_voting.py*, are the 2 driver files for executing various architectures. The former can be used to execute the 3 architectures: 3DCNN, CNN+LSTM (Time Distributed) and CNN-LSTM (ConvLSTM2D), whereas the latter is only for executing majority voting policy using 2D CNNs. Each file during execution requires 2 command line arguuments: `--mode` and   `--model`. Mode denotes whether binary or multiclass classification is to be performed. Model denotes the architecture to be used for training.
- `models/` directory contains 2 files, that contain all the model architectures based on the mode (whether the classification is binary or multiclass).
- `utils/train_utils.py` contains training related functions, `utils/common.py` has functions that are common to both *main.py* and *main_2dcnn_with_majority_voting.py*, `utils/augmentation.py` has augmentation related functions, and `utils/miscellaneous.py` has some other, miscellaneous functions.

## Execution 
Run the code as follows -
```
python main.py --mode binary_or_multiclass --model 3dcnn_or_cnn+lstm_or_convlstm2d
python main_2dcnn_with_majority_voting.py --mode binary_or_multiclass --model 2dcnn
```
Example - 
`python main.py --mode binary --model cnn+lstm`

## License
Copyright (c) 2021 Shreyansh Joshi. Released under the MIT License. See [LICENSE](LICENSE) for details.