A new measure for complexity of decision boundary of DNNs
========
We create the decision boundary complexity (DBC) score to define and measure the complexity of decision boundary of DNNs.

These codes include adversarial sets generation and computations of global DBC and local DBC.

Related paper
======
### Analysis of Generalizability of Deep Neural Networks Based on the Complexity of Decision Boundary

`Citation` S. Guan and M. Loew, "Analysis of Generalizability of Deep Neural Networks Based on the Complexity of Decision Boundary," 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA), 2020, pp. 101-106, doi: 10.1109/ICMLA51294.2020.00025.

[`Paper`](https://doi.org/10.1109/ICMLA51294.2020.00025) [`Arxiv`](https://arxiv.org/abs/2009.07974) [`Video`](https://youtu.be/mJbmPiuGTcU)

Usage
=========
Before running, it needs:
* A two-class classification dataset
* A (DNN) classifier model trained on the given dataset; its outputs are in \[0, 1]

Detailed explainations are in the codes' comments.

Output in `global_DBCs.txt` or `local_DBCs.txt`.
