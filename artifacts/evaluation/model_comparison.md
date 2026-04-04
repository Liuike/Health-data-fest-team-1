# Model Comparison

Conservative comparison of the tuned gradient-boosted tree model (`gbt`) against the logistic regression baseline.

These scores come from a stratified random batch split. They are useful for relative comparison, but they likely overestimate real forecasting performance because nearby rows from the same site can appear in both train and test.

## Category_1

- Preferred on validation F1: `gbt`
- Preferred on test F1: `gbt`
- Validation F1 delta (`gbt - logistic`): `0.1457`
- Test F1 delta (`gbt - logistic`): `0.1150`
- Conservative grade, GBT test: `Fair`
- Conservative grade, logistic test: `Fair`
- Test caution: Optimistic because row-level batch splitting mixes highly similar observations across train and test.

### Validation

| Metric | GBT | Logistic | Delta (GBT - Logistic) | Winner |
| --- | ---: | ---: | ---: | --- |
| auc_pr | 0.9615 | 0.7524 | 0.2092 | gbt |
| roc_auc | 0.9989 | 0.9900 | 0.0089 | gbt |
| precision | 0.9419 | 0.7308 | 0.2111 | gbt |
| recall | 0.9759 | 0.9157 | 0.0602 | gbt |
| f1 | 0.9586 | 0.8128 | 0.1457 | gbt |

### Test

| Metric | GBT | Logistic | Delta (GBT - Logistic) | Winner |
| --- | ---: | ---: | ---: | --- |
| auc_pr | 0.9790 | 0.8473 | 0.1317 | gbt |
| roc_auc | 0.9990 | 0.9921 | 0.0069 | gbt |
| precision | 0.9167 | 0.7422 | 0.1745 | gbt |
| recall | 0.9519 | 0.9135 | 0.0385 | gbt |
| f1 | 0.9340 | 0.8190 | 0.1150 | gbt |

## Category_2

- Preferred on validation F1: `gbt`
- Preferred on test F1: `gbt`
- Validation F1 delta (`gbt - logistic`): `0.0790`
- Test F1 delta (`gbt - logistic`): `0.1724`
- Conservative grade, GBT test: `Fair`
- Conservative grade, logistic test: `Fair`
- Test caution: Optimistic because row-level batch splitting mixes highly similar observations across train and test.

### Validation

| Metric | GBT | Logistic | Delta (GBT - Logistic) | Winner |
| --- | ---: | ---: | ---: | --- |
| auc_pr | 0.9955 | 0.8490 | 0.1466 | gbt |
| roc_auc | 0.9993 | 0.9846 | 0.0148 | gbt |
| precision | 0.9730 | 0.8043 | 0.1686 | gbt |
| recall | 0.9474 | 0.9737 | -0.0263 | logistic_baseline |
| f1 | 0.9600 | 0.8810 | 0.0790 | gbt |

### Test

| Metric | GBT | Logistic | Delta (GBT - Logistic) | Winner |
| --- | ---: | ---: | ---: | --- |
| auc_pr | 0.9753 | 0.7262 | 0.2491 | gbt |
| roc_auc | 0.9971 | 0.9663 | 0.0308 | gbt |
| precision | 0.9200 | 0.6949 | 0.2251 | gbt |
| recall | 0.9583 | 0.8542 | 0.1042 | gbt |
| f1 | 0.9388 | 0.7664 | 0.1724 | gbt |
