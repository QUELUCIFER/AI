使用方法:

--gensim:[gensim-data](https://github.com/RaRe-Technologies/gensim-data)

Example: load a pre-trained model (gloVe word vectors):

        import gensim.downloader as api

        info = api.info()  # show info about available models/datasets
        model = api.load("glove-twitter-25")  # download the model and return as object ready for use
        model.most_similar("cat")

Example: load a corpus and use it to train a Word2Vec model:

        from gensim.models.word2vec import Word2Vec
        import gensim.downloader as api

        corpus = api.load('text8')  # download the corpus and return it opened as an iterable
        model = Word2Vec(corpus)  # train a model from the corpus
        model.most_similar("car")

Example: only download a dataset and return the local file path (no opening):

        import gensim.downloader as api

        print(api.load("20-newsgroups", return_path=True))  # output: /home/user/gensim-data/20-newsgroups/20-newsgroups.gz
        print(api.load("glove-twitter-25", return_path=True))  # output: /home/user/gensim-data/glove-twitter-25/glove-twitter-25.gz

--word2veckeras

        vsk = Word2VecKeras(gensim.models.word2vec.LineSentence('test.txt'),iter=100)
        print( vsk.most_similar('the', topn=5))

        from nltk.corpus import brown
        brk = Word2VecKeras(brown.sents(),iter=10)
        print( brk.most_similar('the', topn=5))
