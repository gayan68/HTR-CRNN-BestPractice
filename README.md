The original work is intrduced by the the paper ["Best Practices for a Handwritten Text Recognition system"](https://arxiv.org/abs/2404.11339)

## My Usage and Contribution
This code was used to see the impact of cross-out words in Handwritten Text recognition (HTR).
I mainly modified the data loaders in the code to pick samples based on probabilities. Also, not all hyper parameters were mentioned in the original code for word level recognition. Therefore, some of the hyper parameters were chosen by a grid seach.

## Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{retsinas2022best,
  title={Best practices for a handwritten text recognition system},
  author={Retsinas, George and Sfikas, Giorgos and Gatos, Basilis and Nikou, Christophoros},
  booktitle={International Workshop on Document Analysis Systems},
  pages={247--259},
  year={2022},
  organization={Springer}
}
```
