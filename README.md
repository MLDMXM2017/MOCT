# MOCT
This repository contains the ​​Python implementation​​ of ​**​MOCT**​ from the paper:

​​"MOCT: A Multi-class Oblique Tree Algorithm for Synergistic Drug Combination Prediction"​

# Requirements
The Python environment and versions of dependency packages used in this implementation are as follows:
```
Python==3.8.18
scikit-learn==1.3.2
numpy==1.24.1
pandas==2.0.3
imbalanced-learn==0.12.4
joblib==1.3.2
```

# Usage
We provide a demo for MOCT in the test.py file, using the HT29, A375 and A549 datasets.
```
python test.py
```

# Hyperparameter Settings
The table below shows the hyperparameters of MOCT​. You can modify these variables to configure the hyperparameters. The settings in test.py represent the ​​default hyperparameter values​​ for MOCT.
| Hyperparameters | Values |
| --- | --- |
| features_num | 50 |
| max_depth | 10 |
| min_samples_leaf | 0.01 |

# Cite This Repository
If you use this code in your research, please cite both the ​​original paper​​ and this ​​code repository​​:
Paper Citation (BibTeX):
```

```
Code Repository
```
@software{MOCT_Code,
  author = {Zhikai Lin},
  title = {MOCT},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/MLDMXM2017/MOCT.git}}
}
```