# Recurrent Soft Attention Model
General Model Structure:

![RSAM structure for 1 timestamp](https://github.com/renll/RSAM/raw/master/111.png)


The weighted context information, i.e. the soft attention, is fed into the model through the down-sampling network that consists of a 1x1 feature-map down-sampling convolutional layer in each glimpse timestamp.

# Related Works
[1] J. Ba, V. Mnih, and K. Kavukcuoglu. Multiple object recognition with visual attention. arXiv67 preprint arXiv:1412.7755, 2014.68

[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectiﬁers: Surpassing human-level69 performance on imagenet classiﬁcation. In Proceedings of the IEEE international conference on70 computer vision, pages 1026–1034, 2015.71 

[3] K.He,X.Zhang,S.Ren,andJ.Sun. Deepresiduallearningforimagerecognition. InProceedings72 of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.73 

[4] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–74 1780, 1997.75 

[5] H. Larochelle and G. E. Hinton. Learning to combine foveal glimpses with a third-order76 boltzmann machine. In Advances in neural information processing systems, pages 1243–1251,77 2010.78 

[6] V. Mnih, N. Heess, A. Graves, et al. Recurrent models of visual attention. In Advances in neural79 information processing systems, pages 2204–2212, 2014.80 

[7] J.Redmon,S.Divvala,R.Girshick,andA.Farhadi. Youonlylookonce: Uniﬁed,real-timeobject81 detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,82 pages 779–788, 2016.83 

[8] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov, R. Zemel, and Y. Bengio.84 Show, attend and tell: Neural image caption generation with visual attention. In International85 Conference on Machine Learning, pages 2048–2057, 201
