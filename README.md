# HTR-CRNN-BestPractice

**HTR-CRNN-BestPractice** is a word-level Handwritten Text Recognition (HTR) model that combines a **Convolutional Recurrent Neural Network (CRNN)** with **Bi-directional LSTM (Bi-LSTM)** layers.  
It is designed to extract features from handwritten word images and capture sequential dependencies for accurate word recognition, following best practices for preprocessing, augmentation, and training.


## Acknowledgment

The original work was introduced in the paper:  
["Best Practices for a Handwritten Text Recognition System"](https://arxiv.org/abs/2404.11339).  

The original codebase is available at:  
[georgeretsi/HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices).  

## Our Usage and Contribution

This repository adapts the original work to study the impact of **cross-out words** in Handwritten Text Recognition (HTR).  

- Modified the data loaders to select samples based on probabilities.  
- Performed hyperparameter tuning (grid search) for word-level recognition, since not all parameters were provided in the original implementation.  
