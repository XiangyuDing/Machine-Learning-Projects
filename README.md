# Machine Learning Projects/《机器学习实战》Python3版本代码

本开源项目包含以下内容：算法的简介，性能分析，基于《机器学习实战》的Python3下的实现，常见错误及解决办法

---

> **第二章：k-近邻算法（KNN）**

- KNN简介：“KNN是分类数据最简单有效的算法，这里通过两个例子讲述了KNN算法构造的分类器。KNN是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据。”

- 优点：
1. KNN的概念足够简单，容易理解（找到离test data最近的k个点，这k个点属于哪个分类更多，则把test data划分为该分类，k一般不超过20，不可以是labels分类数的整数倍）。
2. 这种简单的模型在几何学上非常直观，能够有效解决许多分类问题。

- 缺点：
1. KNN必须保存全部数据集，如果训练数据集很大，必须使用大量的存储空间
2. 必须对数据集中的每个数据计算距离，实际使用可能会非常耗时
3. 它无法给出任何数据的基础结构信息，我们无从知晓平均实例样本和典型实例样本具有哪些特征，也无法得知这些特征之间可能存在的关系

- KNN具体迭代过程可自行Google或参阅：https://www.cnblogs.com/ybjourney/p/4702562.html

- [KNN分类器/数据集：约会对象/28像素数字](https://github.com/XiangyuDing/Machine-Learning-Projects/tree/master/Ch02_KNN)

- [KNN常见错误及解决办法](https://github.com/XiangyuDing/Machine-Learning-Projects/issues/1)

---

> **第三章：决策树（Decision Tree）**

- 决策树简介：“决策树分类器就像带有终止块的流程图，终止块表示分类结果。开始分类前，我们首先测量数据集中特征的不一致性，也就是“熵”，然后寻找最优的分类方案划分数据集，直到数据集中所有的数据集属于同一分类。”

- 两种用来判别分类效果的指标：
1. 基尼不纯度（Gini Impurity）：是指将来自集合中的某种结果随机应用在集合中，某一数据项的预期误差率。是在进行决策树编程的时候，对于混杂程度的预测中，一种度量方式。
2. 信息增益（Information Gain Entropy）：和基尼不纯度类似，熵（信息熵）是另一种衡量集合无序程度的值。熵越大则集合越无序，即划分越失败。

- 缺点：决策树可能会产生过多的数据集划分，从而产生过度匹配数据集的问题。我们可以通过裁剪决策树，合并相邻的无法产生大量信息增益的叶节点，消除匹配过度问题。

- 更多具体公式可自行Google或参阅：http://people.revoledu.com/kardi/tutorial/DecisionTree/how-to-measure-impurity.htm

- [ID3决策树/数据集：戴哪种隐形眼镜？](https://github.com/XiangyuDing/Machine-Learning-Projects/tree/master/Ch03_Decision%20Tree)

- [决策树常见错误及解决办法](https://github.com/XiangyuDing/Machine-Learning-Projects/issues/2)

---

> **第四章：朴素贝叶斯（Naive Bayes）**

- 朴素贝叶斯简介：“对于分类而言，使用概率有时要比使用硬规则更为有效。贝叶斯概率及贝叶斯准则提供了一种利用已知值来估计未知概率的有效方法。”

- 优点：
1. 发源于古典数学理论，有坚实的数学基础，且分类效率稳定。
2. 算法比较简单，对缺失数据也不敏感。

- 缺点：
1. 朴素贝叶斯的假设过于简单，对于文本分类而言，每个单词的出现被假设为独立事件，有过高中程度的语文/英语水平的人都应该知道，这种假设并不成立，单词是相互联系的，如bacon更有可能和delicious连用，而不和garbage连用。
2. 就文本分类问题而言，一个好的词袋模型比起贝叶斯分类器本身更加重要（一个好的先验概率，数据集比分类器更重要）。

- 贝叶斯分类器的原理可自行Google或参阅：http://www.cnblogs.com/pinard/p/6069267.html

- [朴素贝叶斯/数据集：垃圾邮件](https://github.com/XiangyuDing/Machine-Learning-Projects/tree/master/Ch04_Naive%20Bayes)

- [朴素贝叶斯常见错误及解决办法](https://github.com/XiangyuDing/Machine-Learning-Projects/issues/3)
