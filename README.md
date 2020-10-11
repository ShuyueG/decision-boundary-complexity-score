A new measure for complexity of decision boundary of DNNs
========
We create the decision boundary complexity (DBC) score to define and measure the complexity of decision boundary of DNNs.

These codes include adversarial sets generation and computations of global DBC and local DBC.

Related paper
======
### Analysis of Generalizability of Deep Neural Networks Based on the Complexity of Decision Boundary

`In press` International Conference on Machine Learning and Applications (ICMLA), 2020

https://arxiv.org/abs/2009.07974

Usage
=========
Before running, it needs:
* A two-class classification dataset
* A (DNN) classifier model trained on the given dataset; its outputs are in \[0, 1]

Detailed explainations are in the codes' comments.

Output in `global_DBCs.txt` or `local_DBCs.txt`.
