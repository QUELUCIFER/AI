
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




