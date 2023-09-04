# Genetic-Programming-Toolbox
Conventional Genetic Programming framework combined with multiple novel variation operators including "Nearby Neighbors of Semantic Subsets under P" and variation with the Semantically Aware Variational Autoencoder. Regression tasks such as breast cancer detection can be tackled using explainable (human-interpretable) approaches to ML.

This codebase implements a tensor-based GP framework that enables convenient conversion between tree representations and model-based (DL) approaches. It can therefore be used for a variety of future attempts to introduce deep learning into tree-based GP frameworks.
The models folder contains the variation operators (both explicit and (deep) model-based), each of which implements a 'variation' method that allows easy implementation within a conventional GP algorithm such as the one found in evolution.py.

## Usage
Refer to the file 'example_breast_cancer.py' illustrating the configuration and initiation of the algorithms using a breast cancer classification task as an example. In this case a tensor-based implementation of classic GP (baseline) can be compared with the two novel variation operators that provide high locality in symbolic regression (SR) tasks. Refer to 'requirements.txt' for all dependencies. Use 'example_breast_cancer.py' to adjust the configuration, the current config works out-of-the-box.
'''python
python example_breast_cancer.py
'''

