## 目录
- [1.L2正则化](#1l2正则化岭回归)
  - [1.1问题](#11问题)
  - [1.2公式](#12公式)
  - [1.3对应图形](#13对应图形)
  - [1.4使用场景](#14使用场景)
  - [1.5代码实现](https://github.com/mantchs/machine_learning_model/blob/master/Regularization)
- [2.L1正则化lasso回归](#2l1正则化lasso回归)
  - [2.1公式](#21公式)
  - [2.2对应图形](#22对应图形)
  - [2.3使用场景](#23使用场景)
  - [2.4代码实现](https://github.com/mantchs/machine_learning_model/blob/master/Regularization)
- [3.ElasticNet回归](#3elasticnet回归)
  - [3.1公式](#31公式)
  - [3.2使用场景](#32使用场景)
  - [3.3代码实现](#33代码实现)

## 1.L2正则化(岭回归)

### 1.1问题

![](https://images0.cnblogs.com/blog/663864/201411/081949249876584.png)

想要理解什么是正则化，首先我们先来了解上图的方程式。当训练的特征和数据很少时，往往会造成欠拟合的情况，对应的是左边的坐标；而我们想要达到的目的往往是中间的坐标，适当的特征和数据用来训练；但往往现实生活中影响结果的因素是很多的，也就是说会有很多个特征值，所以训练模型的时候往往会造成过拟合的情况，如右边的坐标所示。

### 1.2公式

以图中的公式为例，往往我们得到的模型是：

![UTOOLS1546959038274.png](https://i.loli.net/2019/01/08/5c34b8bda06d3.png)

为了能够得到中间坐标的图形，肯定是**希望θ3和θ4越小越好**，因为这两项越小就越接近于0，就可以得到中间的图形了。

对应的损失函数也加上这个惩罚项(为了惩罚θ)：假设*λ*=1000

![UTOOLS1546959169901.png](https://i.loli.net/2019/01/08/5c34b941f2bf5.png)

为了求得最小值，**使θ值趋近于0**，这就达到了我们的目的，得到中间坐标的方程。

把以上公式通用化得：

![UTOOLS1546959221738.png](https://i.loli.net/2019/01/08/5c34b975cf88d.png)

相当于在原始损失函数中加上了一个惩罚项(λ项)

这就是**防止过拟合**的一个方法，通常叫做**L2正则化，也叫作岭回归。**

### 1.3对应图形

我们可以简化L2正则化的方程：

![UTOOLS1546959273104.png](https://i.loli.net/2019/01/08/5c34b9a91663f.png)

J0表示原始的损失函数，咱们假设正则化项为：

![UTOOLS1546959466689.png](https://i.loli.net/2019/01/08/5c34ba6b35b6a.png)

我们不妨回忆一下圆形的方程：

![UTOOLS1546959496318.png](https://i.loli.net/2019/01/08/5c34ba88553fb.png)

其中(a,b)为圆心坐标，r为半径。那么经过坐标原点的单位元可以写成：

![UTOOLS1546959601400.png](https://i.loli.net/2019/01/08/5c34baf161c01.png)

正和L2正则化项一样，同时，机器学习的任务就是要通过一些方法（比如梯度下降）求出损失函数的最小值。

此时我们的任务变成在L约束下求出J0取最小值的解。

求解J0的过程可以画出等值线。同时L2正则化的函数L也可以在w1w2的二维平面上画出来。如下图：

![UTOOLS1546953455440.png](https://i.loli.net/2019/01/08/5c34a2efa0b01.png)

L表示为图中的黑色圆形，随着梯度下降法的不断逼近，与圆第一次产生交点，而这个交点很难出现在坐标轴上。

这就说明了L2正则化不容易得到稀疏矩阵，同时为了求出损失函数的最小值，使得w1和w2无限接近于0，达到防止过拟合的问题。

### 1.4使用场景

只要数据线性相关，用LinearRegression拟合的不是很好，**需要正则化**，可以考虑使用岭回归(L2), 如何输入特征的维度很高,而且是稀疏线性关系的话， 岭回归就不太合适,考虑使用Lasso回归。

### 1.5代码实现

[GitHub代码--L2正则化](https://github.com/mantchs/machine_learning_model/blob/master/Regularization/RidgeCV.ipynb)

## 2.L1正则化(lasso回归)

### 2.1公式

L1正则化与L2正则化的区别在于惩罚项的不同：

![UTOOLS1546959770276.png](https://i.loli.net/2019/01/08/5c34bb9ac2621.png)

L1正则化表现的是θ的绝对值，变化为上面提到的w1和w2可以表示为：

![UTOOLS1546959815505.png](https://i.loli.net/2019/01/08/5c34bbc779e82.png)

### 2.2对应图形

求解J0的过程可以画出等值线。同时L1正则化的函数也可以在w1w2的二维平面上画出来。如下图：

![UTOOLS1546955675245.png](https://i.loli.net/2019/01/08/5c34ab9a546f7.png)

惩罚项表示为图中的黑色棱形，随着梯度下降法的不断逼近，与棱形第一次产生交点，而这个交点很容易出现在坐标轴上。**这就说明了L1正则化容易得到稀疏矩阵。**

### 2.3使用场景

**L1正则化(Lasso回归)可以使得一些特征的系数变小,甚至还使一些绝对值较小的系数直接变为0**，从而增强模型的泛化能力  。对于高纬的特征数据,尤其是线性关系是稀疏的，就采用L1正则化(Lasso回归),或者是要在一堆特征里面找出主要的特征，那么L1正则化(Lasso回归)更是首选了。

### 2.4代码实现

[GitHub代码--L1正则化](https://github.com/mantchs/machine_learning_model/blob/master/Regularization/LassoCV.ipynb)

## 3.ElasticNet回归

### 3.1公式

**ElasticNet综合了L1正则化项和L2正则化项**，以下是它的公式：

![UTOOLS1546959876945.png](https://i.loli.net/2019/01/08/5c34bc051086c.png)

### 3.2使用场景

ElasticNet在我们发现用Lasso回归太过(太多特征被稀疏为0),而岭回归也正则化的不够(回归系数衰减太慢)的时候，可以考虑使用ElasticNet回归来综合，得到比较好的结果。

### 3.3代码实现

```python
from sklearn import linear_model  
#得到拟合模型，其中x_train,y_train为训练集  
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99],  max_iter=5000).fit(x_train, y_train)  
#利用模型预测，x_test为测试集特征变量  
y_prediction = ENSTest.predict(x_test)
```

.

.

.

![image.png](https://upload-images.jianshu.io/upload_images/13876065-08b587647d14267c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

欢迎添加微信交流！请备注“机器学习”。

