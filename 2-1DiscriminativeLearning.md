###2.1.1 线性回归(Linear Regression)
&emsp;　将样本拟合成连续函数，之前论证过线性回归的目标函数为$h_\theta(x)=\theta^T x+\epsilon$。当目标函数的误差项$ε^{(i)}\sim N(0,\sigma^2)$满足正态分布时，对$P(y│x)$的极大似然估计求解可以推导(推导见笔记)出是对$J(\theta)=\frac{\sum^{m}_{i=1}{(y^i-\theta^T x)^2 }}{2}$求极小，即为最小二乘法(可用梯度下降法求解)。也可以用矩阵计算来求解$J(\theta)$的最小值。
####&emsp;　加权线性回归(LWR)
&emsp;　将每个样本加一个权值(一般权值满足正态分布)，在预测一个未知点$x$时，将$x$代入权值表达式中，得到权值(一般权值范围为从0到1)。然后对加权$J(\theta)$最小化求出系数$\theta^T$，从而得到由未知点周围样本生成的直线。通过此直线来预测未知样本的值。此方法是在预测时动态生成模型。
###2.1.2 逻辑回归(Logisitc Regression)
&emsp;　逻辑回归是用于分类，可以通过指数族推导出$h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}$即在线性回归目标函数上进行函数变换，此变换叫sigma function。具体方法同线性回归，对$P(y│x)$的极大似然估计求解，但是$y$即$h_\theta (x)$满足Bernoulli 概率分布。推导出极大似然函数，然后用梯度下降法求最大值，即$\theta=\theta+\alpha(y-h_\theta(x))x_j$ (与线性回归思路一样，不过线性回归是求$J(\theta)$的最小值)。预测时，因为sigma 函数值是在0和1之间的，所以用0.5为分界。
###2.1.3 感知器学习算法(Perceptron Learning Algorithm)
&emsp;　与逻辑回归类似，用阶梯函数对线性回归的目标函数进行变换，使得新的目标函数$h_\theta(x)=g(\theta^T x)$输出0和1。同样，求极大似然估计，用梯度下降法求最大值。
###2.1.4 多项式分布回归(Softmax Regression)
&emsp;　这里要注意多项式回归与多项式分布回归的区别：多项式回归是在线性回归的基础上，对每一维变量扩展成幂的形式，这样解决了线性不可分的情况。而多项式分布回归是解决的多类分类的问题，即$y$满足Multinomial概率分布。(推导得出参数形式见笔记)
###2.1.5 求极值的算法
&emsp;　在数学中，求极值一般是求一次导数并领其为$\theta$，然后求解未知数。具体到计算机中实现，一般是用梯度下降法或者牛顿法。
####&emsp;　梯度下降法(Gradient Descent)：
&emsp;　选好$\theta$的初始值，然后通过$\theta_j≔\theta_j-\alpha\frac{\partial }{\partial \theta_j} J(\theta)$迭代，其中$theta_j$是第$j$个参数。通过计算可以得到$\theta_j≔\theta_j-\alpha(h_\theta(x)-y)x_j$在迭代部分，有两种方法，一种是批梯度下降法(Batch Gradient Descent)，$\theta_j≔\theta_j-\alpha \sum^{m}_{i=1}{(h_\theta(x^i)-y^i)x_j^i}$计算所有样本后再更新$\theta_j$ 另一种是随机梯度下降(Stochastic Gradient Descent)，一个样本把所有$\theta_j$都更新一边，然后再计算下一个样本。此方法比第一种收敛快，尤其是大样本时比批梯度下降快。梯度下降法依赖于初值，所以容易进入局部最小值。(但是如果本身函数是凸函数，即只有一个最值，那么算法是可以达到全局最优的。)
&emsp;　**数学原理**：由泰勒展开$f(\theta)=f(\theta_0)+(\theta-\theta_0)f^{\prime}(\theta_0)$,可得$\theta-\theta_0=\eta v$(其中，$\eta$标量，$v$单位矢量) 所以$f(\theta)=f(\theta_0)+\eta v f^{\prime}(\theta_0)$。若要求$f$的最小值，即$f(\theta)< f(\theta_0)$ 即$f(\theta)-f(\theta_0)=\eta v f^{\prime}(\theta_0)<0$ 因为$\eta$是标量，所以$v f^{\prime}(\theta_0)<0$ 根据两个向量相乘的公式得到，$vf^{\prime}(\theta_0)=\|v\| \|f^{\prime}(\theta_0)\|cos\alpha$要使$f(\theta)$最小，即$f(\theta)-f(\theta_0)$最大程度的小，所以$cos\alpha=-1$时才可。所以$v$与$f^{\prime}(\theta_0)$方向相反，即$v=-\frac{f^{\prime}(\theta_0)}{\|f^{\prime}(\theta_0)\|}$ 所以$\theta-\theta_0=\eta v$即$\theta=\theta_0+\eta v=\theta_0-\eta\frac{f^{\prime}(\theta_0)}{\|f^{\prime}(\theta_0)\|}$而$\frac{\eta}{\|f^{\prime}(\theta_0)\|}$为一个标量，即上式可以写为$\theta=\theta_0-\eta f^{\prime}(\theta_0)$以上推导只是一维变量，多维类似。
####&emsp;　牛顿法(Newton's Method)：
&emsp;　首先，对于一个函数$f$，如何求$\theta$使$f(\theta)=0$即$\frac{f(\theta)-f(\theta_0)}{(\theta-\theta_0 )}=f^{\prime}(\theta_0)$，因为$f(\theta)=0$，所以上式为$\theta=\theta_0-(f(\theta_0))/(f(\theta_0))$若对极大似然函数求极值，即极大似然函数导数为0，那么$f(\theta)=l^{\prime}(\theta)$，可带入上式。于是可求出θ为何值时极大似然函数值最大。牛顿法优点是迭代速度快，但当θ为向量时，求$l^{\prime\prime}(\theta)$需要计算海森矩阵。