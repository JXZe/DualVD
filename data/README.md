Data
----------------------

1. Download the VisDial v1.0 dialog json files and images from [here][1].
2. Download the word counts file for VisDial v1.0 train split from [here][2]. 
They are used to build the vocabulary.
3. Use Faster-RCNN to extract image features from [here][3].
4. Use Large-Scale-VRD to extract visual relation embedding from [here][4].
5. Use Densecap to extract local captions from [here][5].
6. Generate ELMo word vectors from [here][6].
7. Download pre-trained GloVe word vectors from [here][7].

[1]: https://visualdialog.org/data
[2]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[3]: https://github.com/peteanderson80/bottom-up-attention
[4]: https://github.com/jz462/Large-Scale-VRD.pytorch
[5]: https://github.com/jcjohnson/densecap
[6]: https://allennlp.org/elmo
[7]: https://github.com/stanfordnlp/GloVe
