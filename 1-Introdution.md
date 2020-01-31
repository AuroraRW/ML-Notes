## Introduction
Machine Learning is a very popular word in recent years, not only in computer science but also in other field. So everyone could ask what is Machine Learning? What make it so popular?  

In this part, I will introduce Machine Learning and some concepts about Machine Learning. 
#### *What is Machine Learning?*
Different people has different definition of Machine Learning. In my opinion, Machine Learning is a method which give machine inteligence like people by learning.
![title](1-01.png)
<center>Fig.1</center>  
As the figure1[1] above shows, there are lots of subjects related to Machine Learning, such as Artificial Inteligence(AI), Pattern Recognition, Statistics.   

![title](1-02.jpg)
<center>Fig.2</center> 
We could say that Machine Learning is a part of AI, as the figure2[2] shows. Except the Machine Learning, there are so many parts in AI, like Expert System, Multi-Agent Systems, Evolutionary Computation. About Statistics, Pattern Recognition, there are also intersections between them and Machine Learning. Pattern Recognition is to enable computer to recognize or classify the pattern by itself. Some method of it use Machine Learning algorithm. The different between Statistic and Machine Learning always make us confuse. The bounday between them is not so clear. Statistic ues math theory to build model, so all of these algorithm are supported by math. Machine Learning does not emphasize math theory so much, though some belongs to Statistic Learning.  

Also, we could see from the right part of figure2 that Machine Learning includes Supervised Learning, Unsupervised Learning, Semi-supervised Learning, Ensemble Learning, Deep Learning, and Reinforcement Learning. And the items below connected by dot line are the application of Machine Learning, which are Regression, Classification/Clustering, Outlier(Anomaly) Detection, Metric Learning and Causality Analysis. I will interpret some of these in next part.

In the following part, I will introduce some concepts of Machine Learning.  

#### *Conecpts 1: Supervised Learning vs. Unsupervised Learning vs. Reinforcement Learning*
There are two main kinds of learning in Machine Learning, Supervised Learning and Unsupervised Learning. In Supervised Learning, the labels of samples have been given. The model is trained by using these samples with labels.  Unsupervised Learning is that model is trained by the samples without labels. Reinforcement Learning is a kind of learning by feedback of reward.
#### *Conecpts 2: Parameter Learning vs. Unparameter Learning*

#### *Conecpts 3: Regression vs. Classification*


 

#### Reference
[1]https://www.analyticsvidhya.com/blog/2015/07/difference-machine-learning-statistical-modeling/

[2]https://www.zhihu.com/question/57770020

#5 基本概念

##5.1 监督学习与非监督学习

##5.2 判别学习与生成学习

##5.3 线性模型与非线性模型 （分类具体模型是线性还是非线性）
&emsp;　线性与否不是指目标函数$h_\theta(x)$是否是线性的而是$h_\theta(x)$的参数是否是线性的。即观察x中的每一维看是不是只被一个参数影响，如果是即为线性，如果不是就是非线性。
&emsp;　如，$y=\frac{1}{1+e^{(w_0+w_1*x_1+w_2*x_2 )}}$是线性的，$y=\frac{1}{1+w_5*e^{(w_0+w_1*x_1+w_2*x_2 )}}$是非线性的
##5.4 参数学习与非参数学习
&emsp;　参数学习是模型通过训练求得参数，此参数固定不变。而非参数学习是动态求得参数。
&emsp;　参数学习：
&emsp;　非参数学习：加权线性回归
##5.5 概率模型与非概率模型

##5.6 回归与分类
&emsp;　回归模型离散化后即可得到分类模型。
#6 机器学习算法的评价

$\frac{\partial x}{\partial y}$
$\alpha$
$\sum^{m}_{i=1}{\frac{x}{y}}$
$\displaystyle \sum^{x \to \infty}_{y \to 0}{\frac{x}{y}}$
$\ddot{a}$
$\sim$