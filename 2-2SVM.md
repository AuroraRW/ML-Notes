###2.1.6 支持向量机(Support Vector Machines)
&emsp;　支持向量机被认为是最好的监督学习算法，它可以通过凸优化来达到全局最优，避免神经网络中的容易陷入局部最优的问题。
&emsp;　**第一种情况：线性可分**：记$y\in\{-1,+1\}$，$g(z)=1$当$z\geq0$时，$g(z)=-1$当$z$为其他。则目标函数为$h_{w,b}(x)=g(w^T x+b)$，实际上$w^T x+b$相当于线性回归里的$\theta^Tx$，因为$b$项相当于$\theta_0$。此算法的目的是当线性可分时，在所有样本正确分类的情况下，样本到分类超平面的几何间隔最大。
&emsp;　**函数间隔与几何间隔**：每一个样本关于超平面的函数间隔为$\hat{\gamma}=y^{(i)} (w^T x^{(i)}+b)$。为了获得最大函数间隔，若$y^(i)=1$则需要$w^T x^{(i)}+b\gg0$，若$y^(i)=-1$则需要$w^T x^{(i)}+b\ll0$，若样本的函数间隔$y^{(i)} (w^T x^{(i)}+b)>0$则说明样本正确分类了。整个样本集的函数间隔为最小的样本函数间隔，$\hat{\gamma}=min\hat{\gamma}$。这里函数间隔的意义是保证样本的正确分类。因为$g(w^T x+b)=g(2w^T x+2b)$所以加上一个约束$\|w\|=1$。即${\hat\gamma^{(i)}=y^{(i)}(\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|})$
&emsp;　几何间隔可以度量训练样本与超平面的几何距离。因为$w^Tx+b=0$平面的法向量为$w^T$写成列即为$w$，所以单位法向量为$\frac{w}{\|w\|}$，那么如图所示，$\overrightarrow{B}=\overrightarrow{A}-\gamma^{(i)}\frac{w}{\|w\|}=x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|}$因为$B$在平面上，所以$w^T(x^{(i)}-\gamma^{(i)}\frac{w}{\|w\|})+b=0$，推导得$\gamma^{(i)}=\frac{w^Tx^{(i)}+b}{\|w\|}={\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|}$以上推导为样本为正类时，即$y=1$，若样本为负类时$\gamma$方向相反。所以几何间隔可以定义为${\gamma^{(i)}=y^{(i)}(\frac{w}{\|w\|}}^Tx^{(i)}+\frac{b}{\|w\|})$。因此当约束$\|w\|=1$时，函数间隔和几何间隔相同。同样，整个样本集的几何间隔为所有样本中最小的几何间隔值。
![Alt text](\01.jpg)![Alt text](\02.jpg)
&emsp;　**最大间隔分类器**
&emsp;　$max_{\gamma,w,b}\quad\gamma$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge\gamma, i=1,...,m$
&emsp;　$\qquad \|w\|=1$
&emsp;　目标是求解$w,b$，每个样本都满足上式时(所有样本都能正确分类)，使得$\gamma$最大。约束$\|w\|=1$就是为了让几何间隔与函数间隔相同，即$\frac{\hat\gamma}{\|w\|}=\gamma$所以如上表达可以写为：
&emsp;　$max_{\gamma,w,b}\quad\frac{\hat\gamma}{\|w\|}$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge\hat\gamma, i=1,...,m$
&emsp;　这里因为超平面为$w^T x+b=0$所以同时缩放$w,b$不影响超平面的位置，超平面位置不变，所以几何间隔的值也不变，所以上式还可以用$w,b$表示。此时，可以加约束$\hat\gamma=1$(如右图) 因为如果$\hat\gamma$为1的$m$倍，则$w,b$除以$m$，缩放后位置不变，同时$m$倍的最大值，也是原数的最大值。所以上式可以表达为：
&emsp;　$max_{\gamma,w,b}\quad\frac{1}{\|w\|}$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;  道理同上，也可以继续缩放，将上式转化为$\frac{2}{\|w\|}$的最大值，即为$\frac{\|w\|}{2}$的最小值，因为$\|w\|>0$，所以$\frac{\|w\|}{2}$的最小值也是$\frac{\|w\|^2}{2}$(结合二次函数的图形)。所以上述表达可以写为：
&emsp;  $min_{\gamma,w,b}\quad\frac{1}{2} \|w\|^2$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;　从而得到了如上的最优间隔分类器。
&emsp;　**Lagrange乘数法**
&emsp;　在求解函数的最值时一般都用到Largrange乘数法方法。最优化问题有三种形式: 无约束，只有等式约束，等式约束和不等式约束都有。
&emsp;　第一种，无约束，形式为$min_w f(w)$直接求偏导解方程得出$w$，代入函数中看是否是最值即可。
&emsp;　第二种，只有等式约束，形式为
&emsp;　$min_w f(w)$
&emsp;　s.t. $h_i(w)=0, i=1,...,l$
&emsp;　在求解时写成Largrange函数$L(w,\beta)=f(w)+\sum^l_{i=1}\beta_ih_i(w)$然后求偏导令$\frac{\partial L}{\partial {w_i}}=0,\frac{\partial L}{\partial {\beta_i}}=0$，解出$w,\beta$代入函数中，得到最值。
&emsp;　第三种，等式约束和不等式约束都有
&emsp;　$min_w f(w)$
&emsp;　s.t. $g_i(w)\leq 0$
&emsp;　&emsp; $h_i(w)=0, i=1,...,l$
&emsp;　写成广义Largrange函数$L(w,\alpha,\beta)=f(w)+\sum^k_{i=1}\alpha_ig_i(w)+\sum^l_{i=1}\beta_ih_i(w)$解此类问题时，需要满足KKT条件，然后将原始问题写成对偶问题。
&emsp;　**原始问题和对偶问题**
&emsp;　如上所述的第三种情况的形式叫做原始问题。由于第三种情况的约束比较复杂，所以要想办法把约束去掉。于是对广义Largrange函数$L(x,\alpha,\beta)=f(x)+\sum^k_{i=1}\alpha_ig_i(x)+\sum^l_{i=1}\beta_ih_i(x)$求最大值，对$\alpha,\beta$求最大值，并约定$\alpha_i>0$。在这里$\alpha,\beta$为变量，$x$为常量。所以最大值记为$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)$。下面考虑$x$是否满足约束条件，若不满足，即$g_i(x)>0$或$h_i(x)\neq0$，则$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=+\infty$。因为当$g_i(x)>0$且$\alpha_i>0$，则函数$L$的第二项为无穷大。若$x$满足约束条件，则$L$函数的第三项为0，第二项最大值也为0，而第一项$f(x)$是常量。所以$\theta_p(x)=max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=f(x)$
&emsp;　总结一下，像上一节第三种形式的原始问题，可以写为$min_x f(x)=min_xmax_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=min_x\theta_p(x)$可以记为$p^*$，即$p^*=min_x\theta_p(x)$为原始问题的最优值。
&emsp;　下面再看一下对偶问题，求上面Largrange函数关于$x$的最小值，即$\theta_D(\alpha,\beta)=min_xL(x,\alpha,\beta)$，这是关于$\alpha,\beta$的函数。考虑最大值$max_{\alpha,\beta:\alpha_i\ge0}\theta_D(\alpha,\beta)=max_{\alpha,\beta:\alpha_i\ge0}min_xL(x,\alpha,\beta)$这就是原始问题的对偶问题。形式上与原始问题是对称的。原始问题是先优化$\alpha,\beta$再优化$x$，对偶问题是先优化$x$，再优化$\alpha,\beta$。对偶问题的最优值记为$d^*=max_{\alpha,\beta}\theta_D(\alpha,\beta)$
&emsp;　下面看原始问题与对偶问题的关系。有定理，原始问题的最优值不小于对偶问题的最优值，即$d^*\leq p*$。证明：对任意$\alpha,\beta,x$，有$\theta_D(\alpha,\beta)=min_xL(x,\alpha,\beta)\leq L(x,\alpha,\beta)\leq max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=\theta_p(x)$即$\theta_D(\alpha,\beta)\leq \theta_p(x)$，再对对偶问题和原始问题取最优值，即分别取最大值和最小值，得到$max_{\alpha,\beta:\alpha_i\ge0}\theta_D(\alpha,\beta)\leq min_x\theta_p(x)$，即$d^*\leq p^*$。证明完毕。那么如果等号成立，即原始问题的最优值和对偶问题的最优值相等时，就可以通过求解对偶问题的最优值来得到原始问题的最优值。满足什么样的条件等号成立呢，就是下面要说的KKT条件。
&emsp;　**KKT条件**
&emsp;　首先，如果等号成立即$d*=p*$，即原始问题的最优解和对偶问题的最优解相等，假设最优解为$(x^*,\alpha^*,\beta^*)$。那么此解必然满足原始问题的约束和对偶问题的约束。即$g_i(x^*)\leq0$，$h_i(x^*)=0$，$\alpha_i^*\ge0$,同时，由于存在最优解，所以对变量的偏导数为0，即$\nabla_xL(x^*,\alpha^*,\beta^*)=0，\nabla_{\alpha}L(x^*,\alpha^*,\beta^*)=0，\nabla_{\beta}L(x^*,\alpha^*,\beta^*)=0$。同时，还能得到最重要的一条$\alpha_i^*g_i(x^*)=0$。以上就是KKT条件。下面说明为什么$\alpha_i^*g_i(x^*)=0$。因为$\alpha_i^*\ge0$，若$\alpha_i^*>0$则$g_i(x^*)=0$。因为$g_i(x^*)\leq0$，若$g_i(x^*)<0$，则$\alpha_i^*=0$。就是说$\alpha_i^*$和$g_i(x^*)$必然有一个为0，所以乘积为0。
&emsp;　**线性可分支持向量机的推导**
&emsp;  在之前已经提到了，线性可分支持向量机的问题就是求解如下最优化问题
&emsp;  $min_{\gamma,w,b}\quad\frac{1}{2} \|w\|^2$
&emsp;　s.t. $\quad y^{(i)}(w^T x^{(i)}+b)\ge1, i=1,...,m$
&emsp;  因为上述约束中只有不等式形式，所以Lagrange乘子只有$\alpha_i$，$g_i(w,b)=-y^{(i)}(w^Tx^(i)+b)+1$，由KKT条件可以得到当$\alpha_i>0$时，$g_i(w,b)=0$，即此样本距离分类平面的函数间隔为1。此类样本距离分类平面最近，叫做支持向量。而$\alpha_i=0$的样本是非支持向量。实际上，非支持向量占大多数。参考图2。下面写出Lagrange函数，$L(w,b,\alpha)=\frac{1}{2}\|w\|^2-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)$。整个思路是求解出对偶问题的最优解，根据定义，对偶问题为$\theta_D(\alpha)=min_{w,b}L(w,b,\alpha)$即求Lagrange函数的极值点，然后代入，便得到$\theta_D(\alpha)$。然后再求最大值即为对偶函数的最优值。$\nabla_wL(w,b,\alpha)=w-\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$令其为0，解得$w=\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$同理，$\nabla_bL(w,b,\alpha)=-\sum^m_{i=1}y^{(i)}\alpha_i$令其为0得$\sum^m_{i=1}y^{(i)}\alpha_i=0$然后将$w$带入Lagrange函数得$L(w,b,\alpha)=\frac{1}{2}(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})^T(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)$因为$\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)}$为常数，所以转置等于其本身，即$L(w,b,\alpha)=\frac{1}{2}(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})(\sum^m_{i=1}\alpha_iy^{(i)}x^{(i)})-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)=\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>-\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>+\sum^m_{i=1}\alpha_i$第二项是由$-\sum^m_{i=1}\alpha_i(y^{(i)}(w^Tx^{(i)}+b)-1)=-\sum^m_{i=1}\alpha_iy^{(i)}w^Tx^{(i)}-b\sum^m_{i=1}\alpha_iy^{(i)}+\sum^m_{i=1}\alpha_i$因为$\sum^m_{i=1}\alpha_iy^{(i)}=0$同时将$w$的表达式代入即为所得。于是$L(w,b,\alpha)=\sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i=1}\sum^m_{j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j<x^{{i}},x^{(j)}>$令其为$W(\alpha)$，这其实就是$L(w,b,\alpha)$关于$w,b$的最小值。
&emsp;　下面再考虑$W(\alpha)$关于$\alpha$的最大值


&emsp;　**kernels**

&emsp;　**第二种情况，线性不可分**

&emsp;　几个问题：SVM中的过拟合，小样本问题



