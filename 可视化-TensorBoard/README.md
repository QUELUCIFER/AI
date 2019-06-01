TensorBoard
TensorBoard 是 TensorFlow 上一个非常酷的功能，神经网络很多时候就像是个黑盒子，里面到底是什么样，是什么样的结构，是怎么训练的，可能很难搞清楚。而 TensorBoard 的作用就是可以把复杂的神经网络训练过程给可视化，可以更好地理解，调试并优化程序。 
TensorBoard可以将训练过程中的各种绘制数据展示出来，包括标量（scalars），图片（images），音频（Audio）,计算图（graph）,数据分布，直方图（histograms）和嵌入式向量。

在 scalars 下可以看到 accuracy，cross entropy，dropout，layer1 和 layer2 的 bias 和 weights 等的趋势。 
在 images 和 audio 下可以看到输入的数据。展示训练过程中记录的图像和音频。 
在 graphs 中可以看到模型的结构。 
在 histogram 可以看到 activations，gradients 或者 weights 等变量的每一步的分布，越靠前面就是越新的步数的结果。展示训练过程中记录的数据的分布图 
distribution 和 histogram 是两种不同的形式，可以看到整体的状况。 
在 embedding 中可以看到用 PCA 主成分分析方法将高维数据投影到 3D 空间后的数据的关系。 
Event: 展示训练过程中的统计数据（最值，均值等）变化情况

使用TensorBoard展示数据，需要在执行Tensorflow计算图的过程中，将各种类型的数据汇总并记录到日志文件中。然后使用TensorBoard读取这些日志文件，解析数据并生产数据可视化的Web页面，让我们可以在浏览器中观察各种汇总数据。

keras使用TensorBoard
直接上代码：

        log_filepath = '/tmp/keras_log' 

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

        tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)  
        # 设置log的存储位置，将网络权值以图片格式保持在tensorboard中显示，设置每一个周期计算一次网络的权值，每层输出值的分布直方图 

        cbks = [tb_cb]  

        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))

tensorboard默认的slcar一栏只记录了训练集和验证集上的loss，如何想记录展示其他指标，在model.compile的metric中进行添加，例如：

        model.compile(  
              loss = 'mean_squared_error',  
              optimizer = 'sgd',  
              metrics= c('mae', 'acc')  # 可视化mae和acc 
           )

可视化结果
切换到日志文件路径 /tmp/keras_log 下面，在该文件夹中打开终端，输入：

        tensorboard --logdir=./

如果你在上一级文件夹打开终端，修改对应的路径即可。

笔者在这里遇到Tensorboard报错No dashboards are active for the current data set的问题，大家可以参考文章末尾的参考文献6和7.出现这个问题的主要原因都是日志文件的路径没有写对。最简单的解决方法就是按上述的操作进行，直接在日志文件夹中打开终端进行可视化。

执行完上述命令之后，出现下面的情况。可以看到，直接打开地址http://iotlabk402-HP-Z840-Workstation:6006就可以查看到TensorBoard界面。

原文：https://blog.csdn.net/johinieli/article/details/80070071 

