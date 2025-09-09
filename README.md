# HTR-CRNN-BestPractice

**HTR-CRNN-BestPractice** is a word-level Handwritten Text Recognition (HTR) model that combines a **Convolutional Recurrent Neural Network (CRNN)** with **Bi-directional LSTM (Bi-LSTM)** layers.  
It is designed to extract features from handwritten word images and capture sequential dependencies for accurate word recognition, following best practices for preprocessing, augmentation, and training.


## Acknowledgment

The original work was introduced in the paper:  
["Best Practices for a Handwritten Text Recognition System"](https://arxiv.org/abs/2404.11339).  

The original codebase is available at:  
[georgeretsi/HTR-best-practices](https://github.com/georgeretsi/HTR-best-practices).  

## Data Loaders
The data loaders of this model were modified based on the usage in the paper “A Study of Handwritten Text Recognition with Cross-out Words.”

#### trainer1_word.py
While using trainer1_word.py, it is possible to mix samples from different folders. This must be defined in the config_xxx_xxx.yml file under the probability: section.

For example:
```
probability:
  clean: 0.0
  strike: 1.0
  striked_types: {
    "CLEAN": 0.5,
    "MIXED": 0.0,
    "CLEANED_CLEAN": 0.0,
    "CLEANED_MIXED": 0.0, 
    "SINGLE_LINE": 0.5,
    "DOUBLE_LINE": 0.0,
    "DIAGONAL": 0.0, 
    "CROSS": 0.0,
    "WAVE": 0.0, 
    "ZIG_ZAG": 0.0, 
    "SCRATCH": 0.0,
```

This configuration means the data loader will pick 50% of the samples from the CLEAN folder and 50% from the SINGLE_LINE folder.

#### trainer2_word.py
While using trainer2_word.py, the data picks from the folder spcified in "data --> dataset" in config_xxx_xxx.yml:
```
data:
  dataset: '../DATASETS/IAM/processed_words'
```

#### trainer3_word.py
While using trainer3_word.py, the data is selected based on the probabilities defined in the probability: section of the config_xxx_xxx.yml file.

For example 1:
```
probability:
  clean: 0.0
  # Probabilities must sum to 1
  striked_types: {
    "CLEAN": 0.0,
    "MIXED": 0.0,
    "CLEANED_CLEAN": 0.0,
    "CLEANED_MIXED": 0.0, 
    "SINGLE_LINE": 0.1428,
    "DOUBLE_LINE": 0.1428,
    "DIAGONAL": 0.1428, 
    "CROSS": 0.1428,
    "WAVE": 0.1428, 
    "ZIG_ZAG": 0.1428, 
    "SCRATCH": 0.1428,
```
This configuration means that the dataloader selects random striked samples based on the defined probabilities.


For example 2:
```
probability:
  clean: 0.5
  # Probabilities must sum to 1
  striked_types: {
    "CLEAN": 0.0,
    "MIXED": 0.0,
    "CLEANED_CLEAN": 0.0,
    "CLEANED_MIXED": 0.0, 
    "SINGLE_LINE": 0.1428,
    "DOUBLE_LINE": 0.1428,
    "DIAGONAL": 0.1428, 
    "CROSS": 0.1428,
    "WAVE": 0.1428, 
    "ZIG_ZAG": 0.1428, 
    "SCRATCH": 0.1428,
```
This configuration means that the dataloader picks 50% clean samples from the 'clean_path', and the rest are randomly selected striked samples based on the defined probabilities.

'clean_path' is defined in the config_xxx_xxx.yml file as:
```
data:
  clean_path: '../DATASETS/IAM/processed_words'
```

## Our Usage and Contribution

This repository adapts the original work to study the impact of **cross-out words** in Handwritten Text Recognition (HTR).  

- Modified the data loaders to select samples based on probabilities.  
- Performed hyperparameter tuning (grid search) for word-level recognition, since not all parameters were provided in the original implementation.  
