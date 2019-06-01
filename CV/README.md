## 第一周 卷积神经网络（Foundations of Convolutional Neural Networks）

1.1 计算机视觉（Computer vision）

1.2 边缘检测示例（Edge detection example）

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/47c14f666d56e509a6863e826502bda2.png)

![r](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5f9c10d0986f003e5bd6fa87a9ffe04b.png)

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0c8b5b8441557b671431d515aefa1e8a.png)

1.3 更多边缘检测内容（More edge detection）

1.4 Padding  

![p](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0663e1a9e477e2737067d9e79194208d.png)

1.5 卷积步长（Strided convolutions）

![r](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/16196714c202bb1c8022219394543bf5.png)

![x](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/58810d826a00657957640fb931f792a7.png)

1.6 三维卷积（Convolutions over volumes）  1.7 单层卷积网络（One layer of a convolutional network）

![s](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/53d04d8ee616c7468e5b92da95c0e22b.png)

1.8 简单卷积网络示例（A simple convolution network example）

![v](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0c09c238ff2bcda0ddd9405d1a60b325.png)

![q](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/02028431085fb7974b76156dd4974b68.png)

1.9 池化层（Pooling layers）

![l](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/6bd58a754152e7f5cf55a8c5bbac3100.png)

1.10 卷积神经网络示例（Convolutional neural network example）

![g](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/aa71fe522f85ea932e3797f4fd4f405c.png)

1.11 为什么使用卷积？（Why convolutions?）

![s](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/7503372ab986cd3aedda7674bedfd5f0.png)

## 第二周 深度卷积网络：实例探究（Deep convolutional models: case studies）

2.1 为什么要进行实例探究？（Why look at case studies?）

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/7fb4f4ae7f3dcd200fcaacd5a3188d51.png)

2.2 经典网络（Classic networks）

Lenet5

![l](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5e59b38c9b2942a407b49da84677dae9.png)

Alexnet

![a](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/92575493ecd20003b0b76ac51de0efbb.png)

VGG16

![v](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0a29aeae65a311c56675ad8f1fec2824.png)

2.3 残差网络(ResNets)（Residual Networks (ResNets)）

![r](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/f0a8471f869d8062ba59598c418da7fb.png)

![c](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/6077958a616425d76284cecb43c2f458.png)

2.4 残差网络为什么有用？（Why ResNets work?） skip connection

![r](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/033b59baaa5632368152f5164d17945a.png)

2.5 网络中的网络以及 1×1 卷积（Network in Network and 1×1 convolutions）  network in network

1×1卷积层就是这样实现了一些重要功能的（doing something pretty non-trivial），它给神经网络添加了一个非线性函数，从而减少或保持输入层中的通道数量不变，当然如果你愿意，也可以增加通道数量。

![n](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/46698c486da9ae184532d773716c77e9.png) 

2.6 谷歌 Inception 网络简介（Inception network motivation） 

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/7d160f6eab22e4b9544b28b44da686a6.png)

2.7 Inception 网络（Inception network）

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/16a042a0f2d3866909533d409ff2ce3b.png)

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5315d2dbcc3053b0146cabd79304ef1d.png)

2.8 使用开源的实现方案（Using open-source implementations） [ResNet](https://github.com/KaimingHe/deep-residual-networks)


2.9 迁移学习（Transfer Learning）

如果你有越多的标定的数据，或者越多的Tigger、Misty或者两者都不是的图片，你可以训练越多的层。极端情况下，你可以用下载的权重只作为初始化，用它们来代替随机初始化，接着你可以用梯度下降训练，更新网络所有层的所有权重。

这就是卷积网络训练中的迁移学习，事实上，网上的公开数据集非常庞大，并且你下载的其他人已经训练好几周的权重，已经从数据中学习了很多了，你会发现，对于很多计算机视觉的应用，如果你下载其他人的开源的权重，并用作你问题的初始化，你会做的更好。在所有不同学科中，在所有深度学习不同的应用中，我认为计算机视觉是一个你经常用到迁移学习的领域，除非你有非常非常大的数据集，你可以从头开始训练所有的东西。总之，迁移学习是非常值得你考虑的，除非你有一个极其大的数据集和非常大的计算量预算来从头训练你的网络。

2.10 数据增强（Data augmentation）

2.11 计算机视觉现状（The state of computer vision）

Benchmark 基准测试，Benchmark是一个评价方式，在整个计算机领域有着长期的应用。维基百科上解释：“As computer architecture advanced, it became more difficult to compare the performance of various computer systems simply by looking at their specifications.Therefore, tests were developed that allowed comparison of different architectures.”Benchmark在计算机领域应用最成功的就是性能测试，主要测试负载的执行时间、传输速度、吞吐量、资源占用率等。

## 第三周 目标检测（Object detection）

3.1 目标定位（Object localization）

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/d50ae3ee809da4c728837fee2d055f00.png)

3.2 特征点检测（Landmark detection） 3.3 目标检测（Object detection）

![s](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/ef8afff4e50fc1c50a46b8443f1d6976.png)

比如，将这个窗口（编号1）输入卷积网络，希望卷积网络对该输入区域的输出结果为1，说明网络检测到图上有辆车。

这种算法叫作滑动窗口目标检测，因为我们以某个步幅滑动这些方框窗口遍历整张图片，对这些方形区域进行分类，判断里面有没有汽车。

滑动窗口目标检测算法也有很明显的缺点，就是计算成本，因为你在图片中剪切出太多小方块，卷积网络要一个个地处理。如果你选用的步幅很大，显然会减少输入卷积网络的窗口个数，但是粗糙间隔尺寸可能会影响性能。反之，如果采用小粒度或小步幅，传递给卷积网络的小窗口会特别多，这意味着超高的计算成本。

所以在神经网络兴起之前，人们通常采用更简单的分类器进行对象检测，比如通过采用手工处理工程特征的简单的线性分类器来执行对象检测。至于误差，因为每个分类器的计算成本都很低，它只是一个线性函数，所以滑动窗口目标检测算法表现良好，是个不错的算法。然而，卷积网络运行单个分类人物的成本却高得多，像这样滑动窗口太慢。除非采用超细粒度或极小步幅，否则无法准确定位图片中的对象。

3.4 滑动窗口的卷积实现（Convolutional implementation of sliding windows）

![z](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b33768b46b3a06ff229a153765782b48.png)

3.5 Bounding Box预测（Bounding box predictions）

![yolo](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b6b6ca6167596a180c7bab7296ea850c.png)

3.6 交并比（Intersection over union）

![s](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/38eea69baa46091d516a0b7a33e5379e.png)

3.7 非极大值抑制（Non-max suppression）

![x](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/514cfeb2d7315eba2b6a29f68eae2879.png)

3.8 Anchor Boxes

对象检测中存在的一个问题是每个格子只能检测出一个对象，如果你想让一个格子检测出多个对象，你可以这么做，就是使用anchor box这个概念

![ac](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/e94aa7ea75300ea4692682b179834bb4.png)

![v](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/322b15fe615c739ebd1d36b669748618.png)

3.9 YOLO 算法（Putting it together: YOLO algorithm）  [ You Only Look Once: Unified, Real-Time Object Detection ](https://arxiv.org/abs/1506.02640)

![s](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/66f8cf8e55eadc1ac01f773515bfbc45.png)

最后你要运行一下这个非极大值抑制，为了让内容更有趣一些，我们看看一张新的测试图像，这就是运行非极大值抑制的过程。如果你使用两个anchor box，那么对于9个格子中任何一个都会有两个预测的边界框，其中一个的概率$p_{c}​$很低。但9个格子中，每个都有两个预测的边界框，比如说我们得到的边界框是是这样的，注意有一些边界框可以超出所在格子的高度和宽度（编号1所示）。接下来你抛弃概率很低的预测，去掉这些连神经网络都说，这里很可能什么都没有，所以你需要抛弃这些（编号2所示）。

最后，如果你有三个对象检测类别，你希望检测行人，汽车和摩托车，那么你要做的是，对于每个类别单独运行非极大值抑制，处理预测结果所属类别的边界框，用非极大值抑制来处理行人类别，用非极大值抑制处理车子类别，然后对摩托车类别进行非极大值抑制，运行三次来得到最终的预测结果。所以算法的输出最好能够检测出图像里所有的车子，还有所有的行人（编号3所示）。

3.10 候选区域（选修）（Region proposals (Optional)）

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/e78e4465af892d0965e2b0863263ef8c.png)

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/e6ed1aa3263107d4e189dd75adc060b4.png)

## 第四周 特殊应用：人脸识别和神经风格转换（Special applications: Face recognition &Neural style transfer）

4.1 什么是人脸识别？（What is face recognition?）

![v](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b0d5f91254b48dcc44944bfbdc05992b.png)

4.2 One-Shot学习（One-shot learning）

4.3 Siamese 网络（Siamese network）

上个视频中你学到的函数$d$的作用就是输入两张人脸，然后告诉你它们的相似度。实现这个功能的一个方式就是用Siamese网络

![g](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/214e009729b015bc6088200e3c1ca3cd.png)

怎么训练这个Siamese神经网络呢？不要忘了这两个网络有相同的参数，所以你实际要做的就是训练一个网络，它计算得到的编码可以用于函数$d$，它可以告诉你两张图片是否是同一个人。更准确地说，神经网络的参数定义了一个编码函数$f(x^{(i)})$，如果给定输入图像$x^{(i)}$，这个网络会输出$x^{(i)}$的128维的编码。你要做的就是学习参数，使得如果两个图片$x^{( i)}$和$x^{( j)}$是同一个人，那么你得到的两个编码的距离就小。前面几个幻灯片我都用的是$x^{(1)}$和$x^{( 2)}$，其实训练集里任意一对$x^{(i)}$和$x^{(j)}$都可以。相反，如果$x^{(i)}$和$x^{(j)}$是不同的人，那么你会想让它们之间的编码距离大一点。

如果你改变这个网络所有层的参数，你会得到不同的编码结果，你要做的就是用反向传播来改变这些所有的参数，以确保满足这些条件。

你已经了解了Siamese网络架构，并且知道你想要网络输出什么，即什么是好的编码。但是如何定义实际的目标函数，能够让你的神经网络学习并做到我们刚才讨论的内容呢？在下一个视频里，我们会看到如何用三元组损失函数达到这个目的。

4.4 Triplet 损失（Triplet 损失） [• Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding forFace Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

![t](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/6a701944309f6dce72d03f5070275d5f.png)

![l](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/16bd20003ac6e93b71abb565ac4fd98e.png)

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5d0ec945435eeb29f78463e38c58e90d.png)

总结一下，训练这个三元组损失你需要取你的训练集，然后把它做成很多三元组，这就是一个三元组（编号1），有一个Anchor图片和Positive图片，这两个（Anchor和Positive）是同一个人，还有一张另一个人的Negative图片。这是另一组（编号2），其中Anchor和Positive图片是同一个人，但是Anchor和Negative不是同一个人，等等。

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/347cf0fc665abe47fa0999d8bf771d26.png)

定义了这些包括$A$、$P$和$N$图片的数据集之后，你还需要做的就是用梯度下降最小化我们之前定义的代价函数$J$，这样做的效果就是反向传播到网络中的所有参数来学习到一种编码，使得如果两个图片是同一个人，那么它们的$d$就会很小，如果两个图片不是同一个人，它们的$d$ 就会很大。

这就是三元组损失，并且如何用它来训练网络输出一个好的编码用于人脸识别。现在的人脸识别系统，尤其是大规模的商业人脸识别系统都是在很大的数据集上训练，超过百万图片的数据集并不罕见，一些公司用千万级的图片，还有一些用上亿的图片来训练这些系统。这些是很大的数据集，即使按照现在的标准，这些数据集并不容易获得。幸运的是，一些公司已经训练了这些大型的网络并且上传了模型参数。所以相比于从头训练这些网络，在这一领域，由于这些数据集太大，这一领域的一个实用操作就是下载别人的预训练模型，而不是一切都要从头开始。但是即使你下载了别人的预训练模型，我认为了解怎么训练这些算法也是有用的，以防针对一些应用你需要从头实现这些想法。

4.5 人脸验证与二分类（Face verification and binary classification）

![f](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/2cce4727d7a6d1b1fa10163231d43291.png)

![w](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/bb970476d7de45a473a1c98b8d87b23a.png)

之前提到一个计算技巧可以帮你显著提高部署效果，如果这是一张新图片（编号1），当员工走进门时，希望门可以自动为他们打开，这个（编号2）是在数据库中的图片，不需要每次都计算这些特征（编号6），不需要每次都计算这个嵌入，你可以提前计算好，那么当一个新员工走近时，你可以使用上方的卷积网络来计算这些编码（编号5），然后使用它，和预先计算好的编码进行比较，然后输出预测值$\hat y$。

因为不需要存储原始图像，如果你有一个很大的员工数据库，你不需要为每个员工每次都计算这些编码。这个预先计算的思想，可以节省大量的计算，这个预训练的工作可以用在Siamese网路结构中，将人脸识别当作一个二分类问题，也可以用在学习和使用Triplet loss函数上，我在之前的视频中描述过。

总结一下，把人脸验证当作一个监督学习，创建一个只有成对图片的训练集，不是三个一组，而是成对的图片，目标标签是1表示一对图片是一个人，目标标签是0表示图片中是不同的人。利用不同的成对图片，使用反向传播算法去训练神经网络，训练Siamese神经网络。

4.6 什么是神经风格迁移？（What is neural style transfer?）

4.7 CNN特征可视化（What are deep ConvNets learning?）

4.8 代价函数（Cost function）

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b8dafd082111a86c00066dedd1033ef1.png)

4.9 内容代价函数（Content cost function）

![c](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/2c39492e728ea3605a3247860b23e1a0.png)

![c](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/d54256309adfc1e140390a334bfc49ee.png)

![y](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5b63e8b0c5c991bc19838b709524b79d.png)

现在你需要衡量假如有一个内容图片和一个生成图片他们在内容上的相似度，我们令这个$a^{[l][C]}$和$a^{[l][G]}$，代表这两个图片$C$和$G$的$l$层的激活函数值。如果这两个激活值相似，那么就意味着两个图片的内容相似。

我们定义这个

$J_{\text{content}}( C,G) = \frac{1}{2}|| a^{[l][C]} - a^{[l][G]}||^{2}$

为两个激活值不同或者相似的程度，我们取$l$层的隐含单元的激活值，按元素相减，内容图片的激活值与生成图片相比较，然后取平方，也可以在前面加上归一化或者不加，比如$\frac{1}{2}$或者其他的，都影响不大,因为这都可以由这个超参数$a$来调整（$J(G) =a J_{\text{content}}( C,G) + \beta J_{\text{style}}(S,G)$）。

要清楚我这里用的符号都是展成向量形式的，这个就变成了这一项（$a^{[l]\lbrack C\rbrack}$）减这一项（$a^{[l]\lbrack C\rbrack}$）的$L2$范数的平方，在把他们展成向量后。这就是两个激活值间的差值平方和，这就是两个图片之间$l$层激活值差值的平方和。后面如果对$J(G)$做梯度下降来找$G$的值时，整个代价函数会激励这个算法来找到图像$G$，使得隐含层的激活值和你内容图像的相似。

4.10 风格代价函数（Style cost function）

![m](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/488c5e6bdc2a519b6c620aae53bdf206.png)

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0f0f535ee576716b074ef893097eed44.png)


4.11 一维到三维推广（1D and 3D generalizations of models）

![d](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/49076b88b9ecbd1597f6ae37e8d87dc3.png)

如果下一层卷积使用5×5×5×16维度的过滤器再次卷积，通道数目也与往常一样匹配，如果你有32个过滤器，操作也与之前相同，最终你得到一个6×6×6×32的输出。

某种程度上3D数据也可以使用3D卷积网络学习，这些过滤器实现的功能正是通过你的3D数据进行特征检测。CT医疗扫描是3D数据的一个实例，另一个数据处理的例子是你可以将电影中随时间变化的不同视频切片看作是3D数据，你可以将这个技术用于检测动作及人物行为。


参考文献：

Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering
Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace: Closing the gap to human-level performance in face verification
The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet
Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576)
Harish Narayanan, Convolutional neural networks for artistic style transfer. https://harishnarayanan.org/writing/artistic-style-transfer/
Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style". http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition (https://arxiv.org/pdf/1409.1556.pdf)
MatConvNet. http://www.vlfeat.org/matconvnet/pretrained/

Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - You Only Look Once: Unified, Real-Time Object Detection (2015)
Joseph Redmon, Ali Farhadi - YOLO9000: Better, Faster, Stronger (2016)
Allan Zelener - YAD2K: Yet Another Darknet 2 Keras
The official YOLO website (https://pjreddie.com/darknet/yolo/)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)
Francois Chollet's github repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
