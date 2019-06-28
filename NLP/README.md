1.[NLP-With-Python](https://github.com/susanli2016/NLP-with-Python/)

2.[gensim](https://github.com/RaRe-Technologies/gensim)














### 1.为什么不用one-hot向量标准神经网络？
![pic](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/1653ec3b8eb718ca817d3423ae3ca643.png)
一、是输入和输出数据在不同例子中可以有不同的长度，不是所有的例子都有着同样输入长度$T_{x}$或是同样输出长度的$T_{y}$。即使每个句子都有最大长度，也许你能够填充（pad）或零填充（zero pad）使每个输入语句都达到最大长度，但仍然看起来不是一个好的表达方式。

二、一个像这样单纯的神经网络结构，它并不共享从文本的不同位置上学到的特征。具体来说，如果神经网络已经学习到了在位置1出现的Harry可能是人名的一部分，那么如果Harry出现在其他位置，比如$x^{}$时，它也能够自动识别其为人名的一部分的话，这就很棒了。这可能类似于你在卷积神经网络中看到的，
你希望将部分图片里学到的内容快速推广到图片的其他部分，而我们希望对序列数据也有相似的效果。和你在卷积网络中学到的类似，用一个更好的表达方式也能够让你减少模型中参数的数量。

### 2.RNN
![pic](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/cb041c33b65e17600842ebf87174c4f2.png)


### 3.FP
![pic](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/19cbb2d356a2a6e0f35aa2a946b23a2a.png)


### 4.BP
![pic](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/71a0ed918704f6d35091d8b6d60793e4.png)

![pic](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/rnn_cell_backprop.png)


### 5.word embedding

#### word2vec

![problem](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/776044225ea4a736a4f2b38ea61fae4c.png)

这里有一些解决方案，如分级（hierarchical）的softmax分类器和负采样（Negative Sampling）。

![skip-gram](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/cbow.jpg)  

![cbow](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/skipgram.jpg)

#### GloVe

![q](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/70e282d4d1abb86fd15ff7b175f4e579.png)

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/f6fc2cec52f4ecb15567511aae822914.png)

###2.9 情感分类（Sentiment Classification）

![o](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/ea844a0290e66d1c76a31e34b632dc0c.png)

###2.10 词嵌入除偏（Debiasing Word Embeddings）

![i](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/9b27d865dff73a2f10abbdc1c7fc966b.png)


## 第三周 序列模型和注意力机制（Sequence models & Attention mechanism）

3.3 集束搜索（Beam Search）

3.4 改进集束搜索（Refinements to Beam Search）

![u](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/725eec5b76123bf45c9495e1231b6584.png)

3.5 集束搜索的误差分析（Error analysis in beam search）

![e](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/1bc0b442db9d5a1aa19dfe9a477a3c3e.png)

3.6 Bleu 得分（选修）

![b](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/0f9646d825a0c254376e094b48523ed3.png)

3.8注意力模型（Attention Model）,注意力模型如何让一个神经网络只注意到一部分的输入句子。当它在生成句子的时候，更像人类翻译

![t](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/1e6b86a4e3690b4a0c6b8146ffa2f791.png)

![r](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b22dff4a3b1a4ea8c1ab201446e98889.png)


3.9语音识别（Speech recognition）

音频数据的常见预处理步骤，就是运行这个原始的音频片段，然后生成一个声谱图（a spectrogram），就像这样。同样地，横轴是时间，纵轴是声音的频率（frequencies），而图中不同的颜色，显示了声波能量的大小（the amount of energy），也就是在不同的时间和频率上这些声音有多大。通过这样的声谱图，或者你可能还听过人们谈到过伪空白输出（the false blank outputs），也经常应用于预处理步骤，也就是在音频被输入到学习算法之前，而人耳所做的计算和这个预处理过程非常相似。语音识别方面，最令人振奋的趋势之一就是曾经有一段时间，语音识别系统是用音位（phonemes）来构建的，也就是人工设计的基本单元（hand-engineered basic units of cells），如果用音位来表示"the quick brown fox"，我这里稍微简化一些，"the"含有"th"和"e"的音，而"quick"有"k" "w" "i" "k"的音，语音学家过去把这些音作为声音的基本单元写下来，把这些语音分解成这些基本的声音单元，而"brown"不是一个很正式的音位，因为它的音写起来比较复杂，不过语音学家（linguists）们认为用这些基本的音位单元（basic units of sound called phonemes）来表示音频（audio），是做语音识别最好的办法。不过在end-to-end模型中，我们发现这种音位表示法（phonemes representations）已经不再必要了，而是可以构建一个系统，通过向系统中输入音频片段（audio clip），然后直接输出音频的文本（a transcript），而不需要使用这种人工设计的表示方法。使这种方法成为可能的一件事就是用一个很大的数据集，所以语音识别的研究数据集可能长达300个小时，在学术界，甚至3000小时的文本音频数据集，都被认为是合理的大小。大量的研究，大量的论文所使用的数据集中，有几千种不同的声音，而且，最好的商业系统现在已经训练了超过1万个小时的数据，甚至10万个小时，并且它还会继续变得更大。在文本音频数据集中（Transcribe audio data sets）同时包含$x$和$y$，通过深度学习算法大大推进了语音识别的进程

CTC损失函数（CTC cost）来做语音识别。CTC就是Connectionist Temporal Classification，
算法思想如下:

假设语音片段内容是某人说："the quick brown fox"，这时我们使用一个新的网络，结构像这个样子，这里输入$x$和输出$y$的数量都是一样的，因为我在这里画的，只是一个简单的单向RNN结构。然而在实际中，它有可能是双向的LSTM结构，或者双向的GIU结构，并且通常是很深的模型。但注意一下这里时间步的数量，它非常地大。在语音识别中，通常输入的时间步数量（the number of input time steps）要比输出的时间步的数量（the number of output time steps）多出很多。举个例子，比如你有一段10秒的音频，并且特征（features）是100赫兹的，即每秒有100个样本，于是这段10秒的音频片段就会有1000个输入，就是简单地用100赫兹乘上10秒。所以有1000个输入，但可能你的输出就没有1000个字母了，或者说没有1000个字符。这时要怎么办呢？CTC损失函数允许RNN生成这样的输出：ttt，这是一个特殊的字符，叫做空白符，我们这里用下划线表示，这句话开头的音可表示为h_eee_ _ _，然后这里可能有个空格，我们用这个来表示空格，之后是**_ _ _qqq__，这样的输出也被看做是正确的输出。下面这段输出对应的是"the q"。CTC损失函数的一个基本规则是将空白符之间的重复的字符折叠起来，再说清楚一些，我这里用下划线来表示这个特殊的空白符（a special blank character），它和空格（the space character）是不一样的。所以the和quick之间有一个空格符，所以我要输出一个空格，通过把用空白符所分割的重复的字符折叠起来，然后我们就可以把这段序列折叠成"the q"。这样一来你的神经网络因为有很多这种重复的字符，和很多插入在其中的空白符（blank characters），所以最后我们得到的文本会短上很多。于是这句"the quick brown fox"包括空格一共有19个字符，在这样的情况下，通过允许神经网络有重复的字符和插入空白符使得它能强制输出1000个字符，甚至你可以输出1000个$y$值来表示这段19个字符长的输出。这篇论文来自于Alex Grace**以及刚才提到的那些人。我所参与的深度语音识别系统项目就使用这种思想来构建有效的语音识别系统。

希望这能给你一个粗略的理解，理解语音识别模型是如何工作的：注意力模型是如何工作的，以及CTC模型是如何工作的，以及这两种不同的构建这些系统的方法。现今，在生产技术中，构建一个有效语音识别系统，是一项相当重要的工作，并且它需要很大的数据集，下节视频我想做的是告诉你如何构建一个触发字检测系统（a rigger word detection system），其中的关键字检测系统（keyword detection system）将会更加简单，它可以通过一个更简洁的数量更合理的数据来完成。

3.10触发字检测（Trigger Word Detection）

我这里就简单向你介绍一个你能够使用的算法好了。现在有一个这样的RNN结构，我们要做的就是把一个音频片段（an audio clip）计算出它的声谱图特征（spectrogram features）得到特征向量$x^{<1>}$, $x^{<2>}$, $x^{<3>}$..，然后把它放到RNN中，最后要做的，就是定义我们的目标标签$y$。假如音频片段中的这一点是某人刚刚说完一个触发字，比如"Alexa"，或者"小度你好" 或者"Okay Google"，那么在这一点之前，你就可以在训练集中把目标标签都设为0，然后在这个点之后把目标标签设为1。假如在一段时间之后，触发字又被说了一次，比如是在这个点说的，那么就可以再次在这个点之后把目标标签设为1。这样的标签方案对于RNN来说是可行的，并且确实运行得非常不错。不过该算法一个明显的缺点就是它构建了一个很不平衡的训练集（a very imbalanced training set），0的数量比1多太多了。

这里还有一个解决方法，虽然听起来有点简单粗暴，但确实能使其变得更容易训练。比起只在一个时间步上去输出1，其实你可以在输出变回0之前，多次输出1，或说在固定的一段时间内输出多个1。这样的话，就稍微提高了1与0的比例，这确实有些简单粗暴。在音频片段中，触发字刚被说完之后，就把多个目标标签设为1，这里触发字又被说了一次。说完以后，又让RNN去输出1。在之后的编程练习中，你可以进行更多这样的操作，我想你应该会对自己学会了这么多东西而感到自豪。我们仅仅用了一张幻灯片来描述这种复杂的触发字检测系统。在这个基础上，希望你能够实现一个能有效地让你能够检测出触发字的算法，不过在编程练习中你可以看到更多的学习内容。这就是触发字检测，希望你能对自己感到自豪。因为你已经学了这么多深度学习的内容，现在你可以只用几分钟时间，就能用一张幻灯片来描述触发字能够实现它，并让它发挥作用。你甚至可能在你的家里用触发字系统做一些有趣的事情，比如打开或关闭电器，或者可以改造你的电脑，使得你或者其他人可以用触发字来操作它。




