## Supervised Learning
Supervised Learning includes Discriminative Learning and Generative Learning. In both method, the model is built by learning samples.


&emsp;　判别学习算法是运用数据来学习条件概率分布$P(y|x)$或者直接学习目标函数$h_\theta(x)$算法直接考虑$P(y|x)$的极大似然估计从而求得模型$h_\theta(x)$的参数。在预测阶段，直接用习得的模型进行预测，即利用学习好的目标函数$h_\theta(x)$来进行预测。
&emsp;　生成学习算法运用数据来学习联合概率分布$P(x,y)$，得到各个类别的概率分布模型。算法考虑的是$P(x,y)=P(x|y)P(y)$极大似然估计，从而得到每个类别数据模型$P(x│y)$的参数。在预测阶段，算法是运用贝叶斯公式计算$P(y│x)=\frac{P(x|y)P(y)}{P(x)}$看哪个类别的$P(y│x)$值大即属于哪个类别。所以只能解决分类问题，不能用于回归问题。
#### Discriminative Learning
&emsp;　判别学习算法通过求解$P(y│x)$的极大似然估计来得到模型$h_\theta(x)$的参数，从而直接得到分类模型，即直接求解最优分类面。
&emsp;　**数学原理**：在统计中，大部分概率分布(Gaussian, Multigaussian, Bernoulli, Multinomial, Poisson)都可以写成指数分布族(Exponential Family)的形式，为$P(y,\eta)=b(y)exp⁡(\eta^T T(y)-a(\eta))$。所以判别学习中的概率$P(y│x)$在满足不同分布时也可以写成指数分布族的形式。将具体的分布同时取对数和指数，然后进行变换，即可求出$a,b,T$。这样通过指数分布族可以推导出广义线性模型(GLM, Generalized Linear Model)，满足三条假设。
&emsp;　当$P(y│x)$满足不同分布时，可以推导出不同的$h_\theta(x)$的形式，对应着不同的算法，例如：
&emsp;　满足Bernoulli 概率分布时，此时的$y$为二值(0,1)，可推导出$h_\theta(x)=\frac{1}{1+e^{(-\theta^Tx)}}$即为逻辑回归(Logistic Regression)。 
&emsp;　满足Gaussian概率分布时，此时的$y$为连续变量，$h_\theta(x)=\theta^Tx$为线性回归(Linear Regression)，多项式回归(Polynomial Regression)也是线性回归，为线性组合。
&emsp;　满足Multinomial概率分布时，此时的$y$为多类，$k$个值，$h_\theta(x)$形式见笔记，为softmax回归。
&emsp;　梯度下降法(或牛顿法)：求解$P(y│x)$的极大似然估计来得到模型$h_\theta(x)$的参数时，在概率统计方法里会用导数等于0的方法求极值。在计算机算法中会用到梯度下降法或牛顿法。
&emsp;　可以证明指数族分布的极大似然函数是concave的，所以有最大值。(关于concave和convex的定义见笔记P11)
#### Generative Learning
生成学习算法是对各个类分别建模，通过学习，建立每个类别的$P(x|y)$的概率模型(通过极大似然估计求解模型参数),同时通过样本计算每个类别的$P(y)$，从而得到$P(x,y)=P(x|y)P(y)$。在预测时，利用贝叶斯公式，$P(y=i|x)=\frac{P(x|y=i)P(y)}{P(x)}$,其中$p(x)=\sum_i{P(x|y=i)P(y=i)}$，求得样本$x$对每一类$y=i$的概率，从而得到$x$属于哪一类。
###2.2.1 高斯判别分析(Gaussian discriminant analysis)
&emsp;　高斯判别分析是生成学习算法的一种，是假设每一类$P(x|y=i)$都符合高斯分布。将每一类的密度函数写出，在两类时，参数为$\phi,\mu_0,\mu_1,\Sigma$，利用极大似然估计得出参数，从而得出每个类别的高斯模型。在预测时，由于$P(x)$值与将$x$分为哪一类无关，所以分类时比较$P(y=i|x)$大小，相当于比较$P(x|y=i)P(y)$大小，同时若各类的$y$是均匀分布，则相当于比较各类的$P(x|y=i)$大小。
&emsp;　高斯判别分析与逻辑回归的关系:如果$x|y$满足高斯分布(或者其他分布，如泊松分布)，且$y$为二值两类，则肯定满足逻辑回归。反之不成立。因此，若能肯定模型满足高斯分布，则高斯判别分析要比逻辑回归效果好，若不能肯定，则逻辑回归效果好。
###2.2.2 朴素贝叶斯(Naive Bayes)
&emsp;　此方法的算法思路与其他生成学习算法一样(见2.2)。假设样本之间各维是独立同分布的。同样，由于在预测时与$P(x)$无关，所以只考虑$P(x|y=i)P(y)$
&emsp;　根据样本$x$的每一维满足的概率分布，朴素贝叶斯又有三种模型：伯努利模型，多项式模型，高斯模型。
&emsp;　伯努利模型中，样本的每一维$P(x=j|y=i)$满足伯努利分布(即只要0，1二值)，同时由于各个维是独立同分布的，那么对相当于$P(x_j,y_i)=P(x=j|y=i)P(y=i)=\prod^n _j P(x_j)P(y=i)$的参数进行极大似然估计。在伯努利模型中估计参数$\phi_{j|y=i}$(即每一个$y$的类别中，$x$的各个维出现的概率)以及参数$\phi_{y=i}$(即每一类的概率)
&emsp;　多项式模型，思路和伯努利模型类似。在这里，样本的每一维$P(x=j|y=i)$满足多项式分布(取$k$个值)。
&emsp;　高斯模型，样本的每一维$P(x=j|y=i)$满足高斯分布，即每一维都是连续的，同样进行极大似然估计求出参数(这里注意与高斯判别分析相区别)。另一种处理方法是将连续值离散化。