# CS533: NLP (Spring 2022)
# Course Project 
# Implementation of Goldani et. al. ["Detecting Fake News with Capsule Neural Networks"](https://arxiv.org/pdf/2002.01030v1.pdf).

Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

Forked from Zhao et. al.'s [2018] tensorflow-based capsule network implementation.

Modification made to "networks.py" to include models (short_text_capsule_model() and long_text_capsule_model()) described in Goldani et. al. (2020) [Detecting Fake News with Capsule Neural Networks]. load_LIAR.py and load_ISOT.py contain our code to preprocess the LIAR and ISOT datasets, using stop word removal, tokenization, padding, and glove.6B.300d for creating an embedding matrix. The results are then saved to hdf5 files called ISOT.hdf5 and LIAR.hdf5, which are then passed as shell arguments main.py, modified to incorporate our additions to Zhao et. al.'s code.

# Reference
```
@article{mohammad,
  author    = {Mohammad Hadi Goldani and Saeedeh Momtazi and Reza Safabakhsh},
  title     = {Detecting Fake News with Capsule Neural Networks},
  journal   = {CoRR},
  volume    = {abs/2002.01030},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.01030},
  eprinttype = {arXiv},
  eprint    = {2002.01030},
}

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

