# Datasets

The datasets used in this project are not included in this repository because
they are derived from third-party data sources and may be subject to separate
licenses or terms of use.

Please obtain the data from the original sources, preprocess it into the format
described below, and place the resulting CSV files in this directory. Users are
responsible for ensuring that their use of the data complies with the
applicable licenses and terms.

Expected files:

- `PolitiFact.csv`
- `Snopes.csv`

Expected columns:

- `id_left`
- `cred_label`
- `claim_text`
- `evidence`


## References

If you use the datasets, please cite the original papers:

```bibtex
@inproceedings{popat2017truth,
  title     = {Where the Truth Lies: Explaining the Credibility of Emerging Claims on the Web and Social Media},
  author    = {Kashyap Popat and Subhabrata Mukherjee and Jannik Str{\"{o}}tgen and Gerhard Weikum},
  booktitle = {Proceedings of the 26th International Conference on World Wide Web Companion},
  pages     = {1003--1012},
  year      = {2017},
  publisher = {{ACM}},
  doi       = {10.1145/3041021.3055133}
}
```

```bibtex
@inproceedings{vlachos2014fact,
  title     = {Fact Checking: Task Definition and Dataset Construction},
  author    = {Andreas Vlachos and Sebastian Riedel},
  booktitle = {Proceedings of the Workshop on Language Technologies and Computational Social Science@ACL 2014},
  pages     = {18--22},
  year      = {2014},
  publisher = {Association for Computational Linguistics},
  doi       = {10.3115/V1/W14-2508}
}
```