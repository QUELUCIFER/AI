
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

![skip-gram](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/cbow.jpg)  ![cbow](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/skipgram.jpg)

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

