1.[Python笔记--sklearn函数汇总](https://zhuanlan.zhihu.com/p/35731775)

1. 拆分数据集为训练集和测试集：

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = 
        train_test_split(x, y, test_size = 0.2,random_state=3,shuffle=False)
        # test_size代表测试集的大小，train_size代表训练集的大小，两者只能存在一个
          #random_state代表随即种子编号，默认为None
          #shuffle代表是否进行有放回抽样


2. 数据预处理

2.1 标准化：

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train) # fit生成规则
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.2 区间缩放：

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(x_train) # fit生成规则
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.3 归一化：

        from sklearn.preprocessing import Normalizer
        scaler = Normalizer().fit(x_train) # fit生成规则
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.4 二值化：设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0

        from sklearn.preprocessing import Binarizer
        scaler = Binarizer(threshold=3).fit(x_train) # threshold为设定的阀值
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.5 哑编码处理：

        from sklearn.preprocessing import OneHotEncoder
        scaler = OneHotEncoder().fit(x_train.reshape((-1,1)))
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.6 自定义函数变换

        from numpy import log1p  # 使用对数函数转换
        from sklearn.preprocessing import FunctionTransformer
        scaler = FunctionTransformer(log1p).fit(x_train)
        x_trainScaler = scaler.transform(x_train) # 将规则应用于训练集
        x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

2.7 PCA降维：

        from sklearn.decomposition import PCA
        pca = PCA(n_components=3).fit(x_train)  # n_components设置降维到n维度
        x_trainPca = pca.transform(x_train) # 将规则应用于训练集
        x_testPca = pca.transform(x_test)  # 将规则应用于测试集

2.8 LDA降维：

        from sklearn.lda import LDA
        lda = LDA(n_components=3).fit(x_train)  # n_components设置降维到n维度
        x_trainLda = lda.transform(x_train) # 将规则应用于训练集
        x_testLda = lda.transform(x_test)  # 将规则应用于测试集


3. 特征筛选

3.1 Filter法：

        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=4) -- 用f_classif方法，设定数目为4
        a=selector.fit(x,y)
        print(np.array(a.scores_),'\n',a.get_support())  --  输出得分及选择的结果

3.2 Wrapper法（递归法）：

        from sklearn.linear_model import LinearRegression --导入基模型
        from sklearn.feature_selection import RFE  -- 导入RFE模块
        model1 = LinearRegression()   -- 建立一个线性模型
        rfe = RFE(model1,4)           -- 进行多轮训练，设置筛选特征数目为4个
        rfe = rfe.fit(x,y)            -- 模型的拟合训练
        print(rfe.support_)           -- 输出特征的选择结果
        print(rfe.ranking_)           -- 特征的选择排名


4.构建模型

4.1决策树模型

决策树优点：

便于理解和解释。树的结构可以可视化出来。
训练需要的数据少，对异常值和缺失值不敏感。
能够处理数值型数据和分类数据。
决策树缺点：

容易产生一个过于复杂的模型，使模型产生过拟合的问题。
决策树可能是不稳定的，因为数据中的微小变化可能会导致完全不同的树生成。
        
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='gini', --设置衡量的系数,有entropy和gini，默认gini
                               splitter='best', --选择分类的策略，best和random，默认best
                               max_depth=5, --设置树的最大深度
                               min_samples_split=10,-- 区分一个内部节点需要的最少的样本数
                               min_samples_leaf=5 -- 一个叶节点所需要的最小样本数
                               max_features=5 --最大特征数     
                               max_leaf_nodes=3--最大样本节点个数
                               min_impurity_split --指定信息增益的阀值
                               )
        clf= clf.fit(x_train,y_train)  -- 拟合训练


4.2 逻辑回归模型

优点：实现简单，易于理解和实现；计算代价不高，速度很快，存储资源低。

缺点：容易欠拟合，分类精度可能不高。对异常值和缺失值敏感

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2', --惩罚项（l1与l2），默认l2
                         dual=False, --对偶或原始方法，默认False，样本数量>样本特征的时候，dual通常设置为False
                         tol=0.0001, --停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解
                         C=1.0, --正则化系数λ的倒数，float类型，默认为1.0，越小的数值表示越强的正则化。
                         fit_intercept=True, --是否存在截距或偏差，bool类型，默认为True。
                         intercept_scaling=1, --仅在正则化项为”liblinear”，且fit_intercept设置为True时有用。float类型，默认为1
                         class_weight=None, --用于标示分类模型中各种类型的权重，默认为不输入，也就是不考虑权重，即为None
                         random_state=None, --随机数种子，int类型，可选参数，默认为无
                         solver='liblinear', --优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear
                         max_iter=10, --算法收敛最大迭代次数，int类型，默认为10。
                         multi_class='ovr'--分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。 
                                          如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。 
                         verbose=0, --日志冗长度，int类型。默认为0。就是不输出训练过程
                         warm_start=False, --热启动参数，bool类型。默认为False。
                         n_jobs=1--并行数。int类型，默认为1。1的时候，用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。
                        )
        clf= clf.fit(x_train,y_train)  -- 拟合训练


4.3 线性回归模型

优点：实现简单，可解释性强。

缺点：容易出现欠拟合，对异常值和缺失值比较敏感。

        from sklearn.linear_model import LinearRegression()
        clf = LinearRegression(copy_X=True, 
                       fit_intercept=True, 
                       n_jobs=1, 
                       normalize=False)
        clf= clf.fit(x_train,y_train)  -- 拟合训练


4.4 K-means聚类

        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters=4, --给定的类别数
                    max_iter=100,--为迭代的次数，这里设置最大迭代次数为300
                    n_init=10,--设为10意味着进行10次随机初始化，选择效果最好的一种来作为模型
                    copy_x=True--布尔型，默认值=True,如果把此参数值设为True，则原始数据不会被改变。如果是False，则会直接在原始数据 
                        上做修改并在函数返回值时将其还原。
                    )
        clf= clf.fit(x)  -- 拟合训练

