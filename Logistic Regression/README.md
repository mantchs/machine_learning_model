## 目录

## 1.逻辑回归(Logistic Regression)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[案例](https://github.com/mantchs/machine_learning_model/tree/master/Linear%20Regression/demo)

### 1.1逻辑回归与线性回归的关系

逻辑回归是用来做分类算法的，大家都熟悉线性回归，一般形式是Y=aX+b，y的取值范围是[-∞, +∞]，有这么多取值，怎么进行分类呢？不用担心，伟大的数学家已经为我们找到了一个方法。

首先我们先来看一个函数，这个函数叫做Sigmoid函数：

![](http://www.wailian.work/images/2018/12/10/image061f6.png)

函数中t无论取什么值，其结果都在[0,-1]的区间内，回想一下，一个分类问题就有两种答案，一种是“是”，一种是“否”，那0对应着“否”，1对应着“是”，那又有人问了，你这不是[0,1]的区间吗，怎么会只有0和1呢？这个问题问得好，我们假设分类的阈值是0.5，那么超过0.5的归为1分类，低于0.5的归为0分类，阈值是可以自己设定的。

好了，接下来我们把aX+b带入t中就得到了我们的逻辑回归的一般模型方程：

![](http://www.wailian.work/images/2018/12/10/image4e5fa.png)

结果P也可以理解为概率，换句话说概率大于0.5的属于1分类，概率小于0.5的属于0分类，这就达到了分类的目的。

### 1.2损失函数

逻辑回归的损失函数跟其它的不同，先一睹尊容：

![](http://www.wailian.work/images/2018/12/10/imagedbfb5.png)

解释一下，当真实值为1分类时，用第一个方程来表示损失函数；当真实值为0分类时，用第二个方程来表示损失函数，为什么要加上log函数呢？可以试想一下，当真实样本为1是，但h=0概率，那么log0=∞，这就对模型最大的惩罚力度；当h=1时，那么log1=0，相当于没有惩罚，也就是没有损失，达到最优结果。所以数学家就想出了用log函数来表示损失函数，把上述两式合并起来就是如下函数，并加上正则化项：

![](https://www.wailian.work/images/2018/12/10/image8771f.png)

最后按照梯度下降法一样，求解极小值点，得到想要的模型效果。

### 1.3多分类问题(one vs rest)

其实我们可以从二分类问题过度到多分类问题，思路步骤如下：

1.将类型class1看作正样本，其他类型全部看作负样本，然后我们就可以得到样本标记类型为该类型的概率p1。

2.然后再将另外类型class2看作正样本，其他类型全部看作负样本，同理得到p2。

3.以此循环，我们可以得到该待预测样本的标记类型分别为类型class i时的概率pi，最后我们取pi中最大的那个概率对应的样本标记类型作为我们的待预测样本类型。

![](https://www.wailian.work/images/2018/12/10/image31617.png)

总之还是以二分类来依次划分，并求出概率结果。

### 1.4逻辑回归(LR)的一些经验

- 模型本身并没有好坏之分。
- LR能以概率的形式输出结果，而非只是0,1判定。
- LR的可解释性强，可控度高(你要给老板讲的嘛…)。
- 训练快，feature engineering之后效果赞。
- 因为结果是概率，可以做ranking model。

### 1.5LR的应用

- CTR预估/推荐系统的learning to rank/各种分类场景。
- 某搜索引擎厂的广告CTR预估基线版是LR。
- 某电商搜索排序/广告CTR预估基线版是LR。
- 某电商的购物搭配推荐用了大量LR。
- 某现在一天广告赚1000w+的新闻app排序基线是LR。

### [1.6Python代码实现]()

</br>
</br>
</br>
</br>

![image.png](https://upload-images.jianshu.io/upload_images/13876065-08b587647d14267c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

欢迎添加微信交流！请备注“机器学习”。
