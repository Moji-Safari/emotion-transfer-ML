## 2026-10-07 — Model comparison

Ran compare_models.py. RBF SVM wins (F1 0.719 vs 0.635 for RF).

Question to investigate: Why does RF collapse on S5 (F1=0.338) while
linear SVM gets 0.864? That's backwards. Could be:
  - RF overfitting to other subjects' EDA patterns
  - S5 has unusual baseline EDA levels
  - Something in S5's windowing is different

Also: S14 gives F1=0.000 for logistic, svm_linear, AND random_forest.
Only svm_rbf does slightly better (0.091). S14 may just be hard.

TODO: inspect S14 and S5 individually before proceeding.