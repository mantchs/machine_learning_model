## 目录
- [1.SVM讲解](#1svm讲解案例)
  - [1.1支持向量机(SVM)的由来](#11支持向量机svm的由来)
  - [1.2如何找到超平面](#12如何找到超平面)
  - [1.3最大间隔分类器](#13最大间隔分类器)
  - [1.4后续问题](#14后续问题)
  - [1.5新闻分类实例](https://github.com/mantchs/machine_learning_model/tree/master/SVM/cnews_demo)

## 1.SVM讲解&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[案例](https://github.com/mantchs/machine_learning_model/tree/master/SVM/cnews_demo)

SVM是一个很复杂的算法，不是一篇博文就能够讲完的，所以此篇的定位是初学者能够接受的程度，并且讲的都是SVM的一种思想，通过此篇能够使读着会使用SVM就行，具体SVM的推导过程有一篇博文是讲得非常细的，具体链接我放到最后面，供大家参考。

### 1.1支持向量机(SVM)的由来

首先我们先来看一个3维的平面方程：**Ax+By+Cz+D=0**

这就是我们中学所学的，从这个方程我们可以推导出二维空间的一条直线：**Ax+By+D=0**

那么，依次类推，更高维的空间叫做一个超平面：

![](https://www.wailian.work/images/2018/12/14/image.png)

x代表的是一个向量，接下来我们看下二维空间的几何表示：

![imagea8724.png](https://www.wailian.work/images/2018/12/14/imagea8724.png)

SVM的目标是找到一个超平面，这个超平面能够很好的解决二分类问题，所以先找到各个分类的样本点离这个超平面最近的点，使得这个点到超平面的距离最大化，最近的点就是虚线所画的。由以上超平面公式计算得出大于1的就属于打叉分类，如果小于0的属于圆圈分类。

这些点能够很好地确定一个超平面，而且在几何空间中表示的也是一个向量，**那么就把这些能够用来确定超平面的向量称为支持向量（直接支持超平面的生成），于是该算法就叫做支持向量机(SVM)了。**

### 1.2如何找到超平面

#### 函数间隔

在超平面w*x+b=0确定的情况下，|w*x+b|能够表示点x到距离超平面的远近，而通过观察w*x+b的符号与类标记y的符号是否一致可判断分类是否正确，所以，可以用(y*(w*x+b))的正负性来判定或表示分类的正确性。于此，我们便引出了函数间隔（functional margin）的概念。定义函数间隔（用![image5b81c.png](https://www.wailian.work/images/2018/12/14/image5b81c.png)表示)为：

![image18f21.png](https://www.wailian.work/images/2018/12/14/image18f21.png)

但是这个函数间隔有个问题，就是我成倍的增加w和b的值，则函数值也会跟着成倍增加，但这个超平面没有改变。所以有函数间隔还不够，需要一个几何间隔。

#### 几何间隔

我们把w做一个约束条件，假定对于一个点 x ，令其垂直投影到超平面上的对应点为 x0 ，w 是垂直于超平面的一个向量，为样本x到超平面的距离，如下图所示：

![imageff0e4.png](http://www.wailian.work/images/2018/12/14/imageff0e4.png)

根据平面几何知识，有

![image5b9c1.png](https://www.wailian.work/images/2018/12/14/image5b9c1.png)

其中||w||为w的二阶范数（范数是一个类似于模的表示长度的概念），![imaged559c.png](https://www.wailian.work/images/2018/12/14/imaged559c.png)是单位向量（一个向量除以它的模称之为单位向量）。又由于*x*0 是超平面上的点，满足 *f*(*x*0)=0，代入超平面的方程![image3f5df.png](https://www.wailian.work/images/2018/12/14/image3f5df.png)，可得![imagef9888.png](https://www.wailian.work/images/2018/12/14/imagef9888.png)，即![imagea4151.png](http://www.wailian.work/images/2018/12/14/imagea4151.png)。随即让此式![image5b9c1.png](https://www.wailian.work/images/2018/12/14/image5b9c1.png)的两边同时乘以![image993f1.png](https://www.wailian.work/images/2018/12/14/image993f1.png)，再根据![imagea4151.png](http://www.wailian.work/images/2018/12/14/imagea4151.png)和![image651f2.png](https://www.wailian.work/images/2018/12/14/image651f2.png)，即可算出*γ*：

![image5833a.png](https://www.wailian.work/images/2018/12/14/image5833a.png)

为了得到![image9d8b4.png](https://www.wailian.work/images/2018/12/14/image9d8b4.png)的绝对值，令![image9d8b4.png](https://www.wailian.work/images/2018/12/14/image9d8b4.png)乘上对应的类别 y，即可得出几何间隔（用![image7718c.png](https://www.wailian.work/images/2018/12/14/image7718c.png)表示）的定义：

![image7cee4.png](http://www.wailian.work/images/2018/12/14/image7cee4.png)

### 1.3最大间隔分类器

对一个数据点进行分类，当超平面离数据点的“间隔”越大，分类的确信度（confidence）也越大。所以，为了使得分类的确信度尽量高，需要让所选择的超平面能够最大化这个“间隔”值。这个间隔就是下图中的Gap的一半。

![imagef7dc1.png](https://www.wailian.work/images/2018/12/14/imagef7dc1.png)

回顾下几何间隔的定义![image7cee4.png](http://www.wailian.work/images/2018/12/14/image7cee4.png)，可知：如果令函数间隔![image7718c.png](https://www.wailian.work/images/2018/12/14/image7718c.png)等于1（之所以令![image7718c.png](https://www.wailian.work/images/2018/12/14/image7718c.png)等于1，是为了方便推导和优化，且这样做对目标函数的优化没有影响，至于为什么，请见本文评论下第42楼回复），则有![image7718c.png](https://www.wailian.work/images/2018/12/14/image7718c.png)= 1 / ||w||，从而上述目标函数转化成了

![imageb73e7.png](https://www.wailian.work/images/2018/12/14/imageb73e7.png)

### 1.4后续问题

至此，SVM的第一层已经了解了，就是求最大的几何间隔，对于那些只关心怎么用SVM的朋友便已足够，不必再更进一层深究其更深的原理。

SVM要深入的话有很多内容需要讲到，比如：线性不可分问题、核函数、SMO算法等。

在此推荐一篇博文，这篇博文把深入的SVM内容也讲了，包括推导过程等。如果想进一步了解SVM，推荐看一下：

支持向量机通俗导论：https://blog.csdn.net/v_JULY_v/article/details/7624837#commentBox

### [1.5新闻分类实例](https://github.com/mantchs/machine_learning_model/tree/master/SVM/cnews_demo)

.

.

.

.

![image.png](https://upload-images.jianshu.io/upload_images/13876065-08b587647d14267c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

欢迎添加微信交流！请备注“机器学习”。
