# This repo is for CS533 NLP Course Projects (2022)
# Detecting Fake News with Capsule Neural Networks
Implementation of our paper 
["Detecting Fake News with Capsule Neural Networks"](https://arxiv.org/pdf/2002.01030v1.pdf).

Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

Fake news in the social media has started posing huge problems in the real-world. And numerous fake news detection algorithms has been put forward in the recent years. In this paper, we replicate and reimplement one such paper that makes use of capsule neural networks to enhance the accuracy of fake news identification system. We make use of both static and non-static pre-trained embeddings to capture the word representations. We also apply different levels of n-grams to capture different features. This approach provides 7.8\% and 1\% improvement in the performance over the existing state-of-the-art methods in two well-known datasets, ISOT and LIAR. 

# Reference
If you find our source code useful, please consider citing our work.
```
@inproceedings{zhao2018capsule,
  year = {2018},
  author = {Wei Zhao and Jianbo Ye and Min Yang and Zeyang Lei and Suofei Zhang and Zhou Zhao},
  month = {September},
  title = {Investigating Capsule Networks with Dynamic Routing for Text Classification},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  url = {https://www.aclweb.org/anthology/D18-1350}
}

@inproceedings{zhao2019capsule,
    title = "Towards Scalable and Reliable Capsule Networks for Challenging {NLP} Applications",
    author = "Zhao, Wei and Peng, Haiyun and Eger, Steffen and Cambria, Erik and Yang, Min",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1150",
    doi = "10.18653/v1/P19-1150",
    pages = "1549--1559"
}

@article{DBLP:Zhang2018capsule,
  author    = {Suofei Zhang and Wei Zhao and Xiaofu Wu and Quan Zhou},
  title     = {Fast Dynamic Routing Based on Weighted Kernel Density Estimation},
  journal   = {CoRR},
  volume    = {abs/1805.10807},
  year      = {2018},
  url       = {http://arxiv.org/abs/1805.10807},
  archivePrefix = {arXiv},
  eprint    = {1805.10807},
}
```

