

第00步: 机器学习基础
(可选但强烈推荐)

如果不了解机器学习，可以从Andrew Ng在Coursera上的机器学习公开课开始学习（国内的网易公开课上有Andrew Ng授课时的版本，相对较难一点，但更推荐）。 他在课程里不仅教授了各种各样的机器学习算法，更重要的是，他还详细讲解了一般的机器学习研究方法，如数据预处理、调参等等。

要对深度学习有个初步的认识，可以阅读由Geoff Hinton, Yoshua Bengio, and Yann LeCun等人合著的NIPS 2015 Deep Learning Tutorial ，很适合初学者入门。

第01步: 深入了解深度学习
是的，没有看错，这一步就已经要求我们能深入了解深度学习。

在学习方法上，我更喜欢在网上看深度学习的学习课程的视频，网络上也确实有几门相当不错的课程，下面是我喜欢的几门课程：

Deep learning at Oxford 2015 ，由Nando de Freitas主讲。他详细讲解了深度学习的基础知识，且浅显易懂，适宜入门。当然，如果你已经熟悉了神经网络，迫不及待地想敲开深度学习的大门的话，可以直接从第9节课开始。这门课所有的示例基于深度学习框架Torch。(Videos on Youtube)

Neural Networks for Machine Learning，Geoffrey Hinton 在Coursera上的课程。Hinton是一个极好的研究员，演示了广义的反向传播算法并且对深度学习的发展起到了决定性的作用。我非常非常尊敬他，但不得不说，他的课程有一些欠缺条理，更糟的是Coursera的考试的布置有些混乱。

Neural Networks Class ，Hugo Larochelle主讲的另一门优秀课程。

Yaser Abu-Mostafa’s machine learning course: 本课程更注重理论。

如果你更喜欢看书学习，下面是一些极佳的学习材料。

Neural Networks and Deep Learning Book ，Michael Nielsen著， 这是一本在线书籍并且提供了一些交互式的JavaScript效果。

Deep Learning Book ，由Ian Goodfellow, Yoshua Bengio 和 Aaron Courville合著：内容有些紧凑，但绝对是一本好书。

第10步: 选择一个应用领域并深入了解它
深度学习有很多应用领域，找到一个你感兴趣的，并深入学习。因为深度学习的应用领域非常广泛，所以下面的列表没能全面展示深度学习的所有应用领域。

计算机视觉 
深度学习改写了这一领域。斯坦福大学Andrej Karpathy的CS231n Convolutional Neural Networks for Visual Recognition课程是我所知的最佳学习资料。 他教你基础的计算机视觉知识，并且帮助你建立在AWS上的GPU实例。还可以学习一下Mostafa S. Ibrahim的Getting Started in Computer Vision。

自然语言处理 (NLP) 
自然语言处理领域包含机器翻译、智能问答、情感分析等领域。要掌握此领域需要深入理解算法和自然语言的基础计算属性。Christopher Manning的课程CS 224N / Ling 284是入门的好资料。 CS224d: Deep Learning for Natural Language Processing，是另一门斯坦福公开课，由David Socher (MetaMind创始人)主讲，里面讲解了深度学习与自然处理的最新研究进展，也是一门非常优秀的公开课。要了解更多自然语言处理相关知识的话，可以阅读这篇文章：我是如何学习自然语言处理的？。

记忆网络 (RNN-LSTM) 
结合注意机制的LSTM循环神经网络自外部读写内存的工作意味着在建造系统的时候有一些有意思的工作，就是能够以问答的方式理解、存储和检索信息。这个研究领域起源于NYU的Facebook AI研究室的Yann Lecun博士。Arxiv上最初的博客：Memory Networks。许多的变体、数据集和基准测试都起源于这里，比如，Metamind的自然语言处理的动态记忆网络。

增强学习 
增强学习是通过AlphaGo变火的， 它的围棋系统 最近击败了 历史上围棋最强的选手之一（今年击败柯洁后，应该把可以之一去掉了）。 David Silver的(Google Deepmind) RL视频教程和Stutton研究丰富的教程是一个很好的学习起点。想要了解LSTM的话，可以阅读Christopher的文章理解LSTM 网络 以及Andrej Karpathy的循环神经网络不可思议的力量。

生成模型 
尽管有辨识力的模型会尝试检测、识别和分割事物，但它们最终在倒在了寻找特征的路上并且不能理解数据的基本层面。除了短期应用之外，生成模型还提供自动学习自然特征的可能性，类别、尺寸或者完全是其它的特征。下面是三种常用的生成模型 —— 生成对抗网络 (GANs)，变分自编码器(VAEs) 和 对抗模型 (比如 PixelRNN)。其中，GAN是最流行的。深入阅读：

Ian Goodfellow et al.最初的GAN论文

拉普拉斯对抗网络(LAPGAN)论文 可用于修复稳定性

深度卷积生成对抗网络(DCGAN)论文 和DCGAN代码 可以用来在没有监督的情况下学习分层的特征。同时, 可以看看DCGAN图像超分辨

第11步: 知行合一
动手做项目是成为专家的关键。尝试去做一些你感兴趣而且和你技能水平相匹配的项目。下面是几个帮助你思考的建议：

传统学习方法之一，在MNIST dataset上学习分类

在ImageNet上进行人脸识别和分类。有兴趣的话，可以参加ImageNet Challenge 2016。

使用RNNs 和 CNNs在推特上进行情感分析（国内的话可以分析微博啊，贴吧啊等）

教神经网络学习著名艺术家的艺术风格， (艺术风格的神经元算法)

使用循环神经网络作曲 
，使用循环神经网络演奏ping-pong

使用神经网络评价自拍

使用深度学习自动地将图片变为黑白色

想要有更多的灵感的话, 就看看CS231n Winter 2016 和 Winter 2015 这两个项目吧。同时还可以关注一下Kaggle和HackerRank竞赛，可以以个人或团队形式在里面进行参赛，竞赛既可以帮助你学习，实力够的话，也可以去获取奖金。

其他资源
下面是一些帮助你继续学习的方法：

阅读优秀的Blog。包括Christopher Olah的博客 和 Andrew Karpathy的博客，这两个博客上有基础概念解析文章和最新研究成果的跟踪文章。

关注Twitter上那些有影响力的学者。 这里是一部分： @drfeifei, @ylecun, @karpathy, @AndrewYNg, @Kdnuggets, @OpenAI, @googleresearch. (更多: Who to follow on Twitter for machine learning information ? )

Yann LecunnGoogle+ 深度学习公共网页, 这是一个接触深度学习创新成果以及与专业研究员、深度学习爱好者交流的好去处。

Github上ChristosChristofidis/awesome-deep-learning项目里有很多深度学习的教程、项目，还有有趣的社区，值得star和fork。
