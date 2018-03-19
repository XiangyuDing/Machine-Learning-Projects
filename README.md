# Machine Learning Projects/《机器学习实战》Python3版本代码
包含常见错误及解决办法：

> **第二章：k-近邻算法（KNN）**

- KNN简介：“KNN是分类数据最简单有效的算法，这里通过两个例子讲述了KNN算法构造的分类器。KNN是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据。”

- 优点：
1. KNN的概念足够简单，容易理解（找到离test data最近的k个点，这k个点属于哪个分类更多，则把test data划分为该分类）。
2. 这种简单的模型在几何学上非常直观，能够有效解决许多分类问题。

- 缺点：
1. KNN必须保存全部数据集，如果训练数据集很大，必须使用大量的存储空间
2. 必须对数据集中的每个数据计算距离，实际使用可能会非常耗时
3. 它无法给出任何数据的基础结构信息，我们无从知晓平均实例样本和典型实例样本具有哪些特征，也无法得知这些特征之间可能存在的关系

- [KNN分类器/数据集](https://github.com/XiangyuDing/Machine-Learning-Projects/tree/master/KNN)

- [KNN常见错误及解决办法](https://github.com/XiangyuDing/Machine-Learning-Projects/issues/1)

> **第三章：决策树**

- 决策树简介：“决策树分类器就像带有终止块的流程图，终止块表示分类结果。开始分类前，我们首先测量数据集中特征的不一致性，也就是“熵”，然后寻找最优的分类方案划分数据集，直到数据集中所有的数据集属于同一分类。”

- 缺点：决策树可能会产生过多的数据集划分，从而产生过度匹配数据集的问题。我们可以通过裁剪决策树，合并相邻的无法产生大量信息增益的叶节点，消除匹配过度问题。

- [决策树/数据集](https://github.com/XiangyuDing/Machine-Learning-Projects/tree/master/Ch03_Decision%20Tree)

- [决策树常见错误及解决办法](https://github.com/XiangyuDing/Machine-Learning-Projects/issues/2)
