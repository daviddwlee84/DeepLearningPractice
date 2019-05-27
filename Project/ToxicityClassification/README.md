# Jigsaw Unintended Bias in Toxicity Classification

* PKU Course team member's github repository: [WyAzx/ml_final_project](https://github.com/WyAzx/ml_final_project)

## Kaggle Competition

Getting started

```sh
# Download the dataset
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
```

### Dataset

### Submission

For evaluation, test set examples with target >= 0.5 will be considered to be in the positive class (toxic).

> Models do not need to predict the additional attributes for the competition

```csv
id,prediction
7000000,0.0
7000001,0.0
etc.
```

### Evaluation

Submetric

* Overall AUC: the ROC-AUC for the full evaluation set
* Bias AUCs:
  * Subgroup AUC
  * BPSN (Background Positive, Subgroup Negative) AUC
  * BNSP (Background Negative, Subgroup Positive) AUC

> Generalized Mean of Bias AUCs
>
>     $$
>     M_p(m_s) = \left(\frac{1}{N} \sum_{s=1}^{N} m_s^p\right)^\frac{1}{p}
>     $$

#### Final Metric

$$
score = w_0 AUC_{overall} + \sum_{a=1}^{A} w_a M_p(m_{s,a})
$$

## Resources

### Popular Kernel

Preprocessing

* [Simple EDA Text Preprocessing - Jigsaw](https://www.kaggle.com/nz0722/simple-eda-text-preprocessing-jigsaw)
* [[Public Version] Text Cleaning - Vocab ~65%+](https://www.kaggle.com/adityaecdrid/public-version-text-cleaning-vocab-65)

Model

* [Simple LSTM](https://www.kaggle.com/thousandvoices/simple-lstm) - 93%
