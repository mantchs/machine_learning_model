## 1.信用卡欺诈预测案例

我们都知道信用卡，能够透支一大笔钱来供自己消费，正因为这一点，不法分子就利用信用卡进一特性来实施欺诈行为。银行为了能够检测出这一欺诈行为，通过机器学习模型进行智能识别，提前冻结该账户，避免造成银行的损失。那么我们应该通过什么方式来提高这种识别精度呢！这就是今天要说的主题，多模型融合预测。使用到的模型算法有：**KNN、SVM、Logistic Regression(LR)、Random Forest**。

我会讲到**如何使用多模型进行融合计算(模型集成)、模型评估、超参数调节、K折交叉验证**等，力求能够讲得清楚，希望大家通过这篇博文能够了解到一个完整的机器学习算法到底是怎样的，如有讲得不到位亦或是错误的地方，望告知！

以下我们正式开始介绍。

**数据集下载：**https://v2.fangcloud.com/share/a63342d8bd816c43f281dab455

**GitHub完整代码：**

## 2.模型集成(model ensemble)

我们先从概念着手，这是我们的地基，要建起高楼大厦，首先地基要稳。

- **多模型：**分类问题是以多个模型计算出的结果进行投票决定最终答案，线性问题以多个模型计算出来的结果求取均值作为预测数值。

那么多模型融合存在着多种实现方法：**Bagging思想、Stacking、Adaboost。**

### 2.1Bagging

Bagging是bootstrap aggregating。Bagging思想就是从总体样本当中随机取一部分样本进行训练，通过多次这样的结果，进行投票亦或求取平均值作为结果输出，这就极大可能的避免了不好的样本数据，从而提高准确度。因为有些是不好的样本，相当于噪声，模型学入噪声后会使准确度不高。一句话概括就是：**群众的力量是伟大的，集体智慧是惊人的。**

而反观多模型，其实也是一样的，利用多个模型的结果进行投票亦或求取均值作为最终的输出，用的就是Bagging的思想。

### 2.2Stacking

stacking是一种分层模型集成框架。以两层为例，第一层由多个基学习器组成，其输入为原始训练集，第二层的模型则是以第一层基学习器的输出作为训练集进行再训练，从而得到完整的stacking模型。如果是多层次的话，以此类推。一句话概括：**站在巨人的肩膀上，能看得更远。**

![TIM截图20181231134358.png](https://i.loli.net/2018/12/31/5c29aca8a6f0e.png)

### 2.3Adaboost

所谓的AdaBoost的核心思想其实是，既然找一个强分类器不容易，那么我们干脆就不找了吧！我们可以去找多个弱分类器，这是比较容易实现的一件事情，然后再集成这些弱分类器就有可能达到强分类器的效果了，其中这里的弱分类器真的是很弱，你只需要构建一个比瞎猜的效果好一点点的分类器就可以了。一句话概括：**坚守一万小时定律，努力学习。**

### 2.4图解模型集成

![无标题.png](https://i.loli.net/2018/12/31/5c29b3ef23dc1.png)

## 3.案例总流程

![未命名文件 (1).jpg](https://i.loli.net/2018/12/31/5c29bf94e3484.jpg)

1. 首先拉取数据到python中。
2. 将数据划分成训练集和测试集，训练集由于分类极度不平衡，所以采取下采样工作，使分类比例达到一致。
3. 将训练集送入模型中训练，同时以K折交叉验证方法来进行超参数调节，哪一组超参数表现好，就选择哪一组超参数。
4. 寻找到超参数后，用同样的方法寻找决策边界，至此模型训练完成。
5. 使用模型集成预测测试集，并使用ROC曲线分析法，得到模型的评估指标。

## 4.初始化工作

啥都不说，先上代码，这里需要说明的就是sklearn.model_selection这个类库，因为老版本和新版本的区别还是很大的，如果巡行报错，尝试着升级sklearn库。

```python
# 数据读取与计算
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理与模型选择
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import itertools

# 随机森林与SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# 一些基本参数设定
mode = 2        #投票个数阈值
ratio = 1       #负样本倍率
iteration1 = 1  #总流程循环次数
show_best_c = True  #是否显示最优超参数
show_bdry = True    #是否显示决策边界

##读取数据,删除无用的时间特征。
data=pd.read_csv('creditcard.csv')
data.drop('Time',axis=1,inplace=True)
```



## 5.数据下采样

先回答什么是数据下采样：

**数据下采样：**数据集中正样本和负样本的比例严重失调，这会给模型的学习带来很大的困扰，例如，正样本有100个，而负样本只有1个，模型只是看到了正样本，而学习不到负样本，这回造成模型对负样本的预测能力几乎为0。所以为了避免这种数据倾斜，处理数据使得正样本和负样本的数量基本均等，这样的模型泛化能力才会高。

反观**数据上采样**也是一样的，只不过是基准样本不一样而已。

这里的数据处理采用下标的方式，较容易运算。

```python
#欺诈类的样本下标
fraud_indices=np.array(data[data.Class==1].index)
#进行随机排列
np.random.shuffle(fraud_indices)

#获取正常样本下标
normal_indices=np.array(data[data.Class==0].index)
np.random.shuffle(normal_indices)


#划分训练集和测试集
train_normal_indices, train_fraud_indices, test_normal_indices
	,test_fraud_indices = split_train_test(normal_indices,fraud_indices)

##合并测试集
test_indices=np.concatenate([test_normal_indices,test_fraud_indices])

#通过下标选取测试集数据，[表示选取行,表示选取列]
test_data=data.iloc[test_indices,:]
x_test=test_data.ix[:,test_data.columns != 'Class']
y_test=test_data.ix[:,test_data.columns == 'Class']

#数据下采样，调用下采样函数 getTrainingSample
x_train_undersample,y_train_undersample,train_normal_pos=getTrainingSample(
train_fraud_indices,train_normal_indices,data,0,ratio)
```

getTrainingSample函数如下，由于代码显示效果不行，所以以图代替，文章开头已有源代码链接，注解已写得很清楚，不需重复赘述：

![UTOOLS1546243337995.png](https://i.loli.net/2018/12/31/5c29cd0a797ff.png)



## 6.模型训练

### 6.1KNN

```python
#用不同的模型进行训练
models_dict = {'knn' : knn_module, 'svm_rbf': svm_rbf_module, 'svm_poly': svm_poly_module,
'lr': lr_module, 'rf': rf_module}

#knn中取不同的k值(超参数)
c_param_range_knn=[3,5,7,9]
#自定义cross_validation_recall，使用循环找出最适合的超参数。
best_c_knn=cross_validation_recall(x,y, c_param_range_knn,models_dict, 'knn')
```

cross_validation_recall函数如下：

![UTOOLS1546245831285.png](https://i.loli.net/2018/12/31/5c29d6c7816f9.png)

这里有几个概念需要解释一下，以防大家看不懂。

- **K折交叉验证：**K折交叉验证(k-fold cross-validation)首先将所有数据分割成K个子样本，不重复的选取其中一个子样本作为测试集，其他K-1个样本用来训练。共重复K次，平均K次的结果或者使用其它指标，最终得到一个单一估测。

  这个方法的优势在于，保证每个子样本都参与训练且都被测试，降低泛化误差。其中，10折交叉验证是最常用的。

- **ROC曲线**：评估模型好坏的方式，已有人解释非常清楚，此处不再赘述，欲了解请点击：

  https://www.cnblogs.com/gatherstars/p/6084696.html

接下来就是真正的模型训练函数：

```python
def knn_module(x,y,indices, c_param, bdry=None):
    #超参数赋值
    knn=KNeighborsClassifier(n_neighbors=c_param)
    #ravel把数组变平
    knn.fit(x.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = knn.predict(x.iloc[indices[1],:].values)
    
    return y_pred_undersample
```

模型评估，计算召回率和auc值：

```python
#计算召回率和auc
#y_t是真实值，y_p是预测值
def compute_recall_and_auc(y_t, y_p):
    #混淆矩阵 https://www.cnblogs.com/zhixingheyi/p/8097782.html
    #  https://blog.csdn.net/xierhacker/article/details/70903617
    cnf_matrix=confusion_matrix(y_t,y_p)
    #设置numpy的打印精度
    np.set_printoptions(precision=2)
    recall_score = cnf_matrix[0,0]/(cnf_matrix[1,0]+cnf_matrix[0,0])
    
    #Roc曲线
    # https://www.cnblogs.com/gatherstars/p/6084696.html
    fpr, tpr,thresholds = roc_curve(y_t,y_p)
    roc_auc= auc(fpr,tpr)
    return recall_score , roc_auc
```



### 6.2  SVM-RBF

径向基函数（RBF）做SVM的核函数。

欲想了解核函数：https://blog.csdn.net/v_JULY_v/article/details/7624837#commentBox

```python
# SVM-RBF中不同的参数
c_param_range_svm_rbf=[0.01,0.1,1,10,100]
best_c_svm_rbf = cross_validation_recall(x,y,c_param_range_svm_rbf, models_dict, 'svm_rbf')

def svm_rbf_module(x, y, indices, c_param, bdry= 0.5):
    svm_rbf = SVC(C=c_param, probability=True)
    svm_rbf.fit(x.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_rbf.predict_proba(x.iloc[indices[1],:].values)[:,1] >= bdry#True/Flase
    return y_pred_undersample
```

### 6.3 SVM-POLY

多项式（POLY）做SVM的核函数。

![UTOOLS1546247241520.png](https://i.loli.net/2018/12/31/5c29dc498aea3.png)

训练函数为：

```python
def svm_poly_module(x,y, indices, c_param, bdry=0.5):
    svm_poly=SVC(C=c_param[0], kernel='poly', degree= c_param[1], probability=True)
    svm_poly.fit(x.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample = svm_poly.predict_proba(x.iloc[indices[1],:].values)[:,1] >= bdry
    return y_pred_undersample
```

### 6.4 Logistic Regression

逻辑回归模型

```python
# 逻辑回归当中的正则化强度
c_param_range_lr=[0.01,0.1,1,10,100]
best_c_lr = cross_validation_recall(x,y, c_param_range_lr, models_dict, 'lr')

def lr_module(x,y, indices, c_param, bdry=0.5):
    # penalty惩罚系数
    lr = LogisticRegression(C=c_param,penalty='11')
    lr.fit(X.iloc[indices[0],:], y.iloc[indices[0],:].values.ravel())
    y_pred_undersample= lr.predict_proba(X.iloc[indices[1],:].values)[:,1]>=bdry
    return y_pred_undersample
```

### 6.5 Random Forest

随机森林模型，欲知超参数含义请点击：

https://www.cnblogs.com/harvey888/p/6512312.html

```python
# 随机森林里调参
c_param_range_rf = [2,5,10,15,20]
best_c_rf= cross_validation_recall(X, y, c_param_range_rf, models_dict, 'rf')
```

![UTOOLS1546247662478.png](https://i.loli.net/2018/12/31/5c29ddee81187.png)



### 6.6决策边界

在具有两个类的统计分类问题中，决策边界或决策表面是超曲面，其将基础向量空间划分为两个集合，一个集合。 分类器将决策边界一侧的所有点分类为属于一个类，而将另一侧的所有点分类为属于另一个类。

所以这一步我们要做的就是根据AUC值找出模型最好的决策边界值，也就是概率值。大于这一概率值为正样本，反之为负样本。

```python
# 交叉验证确定合适的决策边界阈值
fold = KFold(4,shuffle=True)

# 定义各个模型的计算公式
def lr_bdry_module(recall_acc, roc_auc):
    return 0.9*recall_acc+0.1*roc_auc
def svm_rbf_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def svm_poly_bdry_module(recall_acc, roc_auc):
    return recall_acc*roc_auc
def rf_bdry_module(recall_acc, roc_auc):
    return 0.5*recall_acc+0.5*roc_auc
bdry_dict = {'lr': lr_bdry_module,'svm_rbf': svm_rbf_bdry_module,
             'svm_poly': svm_poly_bdry_module, 'rf': rf_bdry_module}

# decision_boundary是一个计算决策边界的函数
best_bdry_svm_rbf= decision_boundary(x, y, fold, best_c_svm_rbf, bdry_dict, models_dict, 'svm_rbf')
best_bdry_svm_poly = decision_boundary(x, y, fold, best_c_svm_poly, bdry_dict, models_dict, 						'svm_poly')
best_bdry_lr = decision_boundary(x, y, fold, best_c_lr, bdry_dict, models_dict, 'lr')
best_bdry_rf = decision_boundary(x, y, fold, best_c_rf, bdry_dict, models_dict, 'rf')
best_bdry = [0.5, best_bdry_svm_rbf, best_bdry_svm_poly, best_bdry_lr, best_bdry_rf]
```

decision_boundary函数为，与前面寻找超参数大致相同：

![UTOOLS1546248292832.png](https://i.loli.net/2018/12/31/5c29e06617b39.png)



### 6.7 模型建模

寻找到最优的超参数和决策边界后，就可以正式开始训练各个模型了。

```python
# 最优参数建模
knn = KNeighborsClassifier(n_neighbors = int(best_c_knn))
knn.fit(x.values, y.values.ravel())

svm_rbf = SVC(C=best_c_svm_rbf, probability = True)
svm_rbf.fit(x.values, y.values.ravel())

svm_poly = SVC(C=best_c_svm_poly[0], kernel = 'poly', degree = best_c_svm_poly[1], probability = True)
svm_poly.fit(x.values, y.values.ravel())

lr = LogisticRegression(C = best_c_lr, penalty ='l1', warm_start = False)
lr.fit(x.values, y.values.ravel())

rf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, criterion = 'entropy', 
max_features = 'auto', max_depth = None, 
min_samples_split  = int(best_c_rf), random_state=0)
rf.fit(x.values, y.values.ravel())

models = [knn,svm_rbf,svm_poly, lr, rf]
```



## 7.结果

### 7.1预测

使用之前划分的测试集运用以上训练出来的模型进行预测，预测使用的是模型集成的投票机制。

我们先来看看预测的代码：

![UTOOLS1546248653719.png](https://i.loli.net/2018/12/31/5c29e1cdbd6cc.png)

模型集成投票代码：

![UTOOLS1546248915382.png](https://i.loli.net/2018/12/31/5c29e2d3aa3f2.png)

### 7.2模型评估

使用AUC进行模型评估，预测部分代码已经记录有相关指标数据，只要计算平均得分就可以。

```python
#计算平均得分
mean_recall_score = np.mean(recall_score_list)
std_recall_score = np.std(recall_score_list)

mean_auc= np.mean(auc_list)
std_auc = np.std(auc_list)
```



## 8.完整代码

**数据集下载：**https://v2.fangcloud.com/share/a63342d8bd816c43f281dab455

**GitHub完整代码：**

.

.

.

![image.png](https://upload-images.jianshu.io/upload_images/13876065-08b587647d14267c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

欢迎添加微信交流！请备注“机器学习”。
