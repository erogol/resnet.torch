<h3>resnet.torch</h3>

This is a fork of https://github.com/facebook/fb.resnet.torch. Refer to that if you need to know the details of this library.

This code is heavily modified with many additions throughout my research. Many of the changes are optional and defined in "opts.lua". Here is the list of the additions by no means complete.

1. Class weighting to tackle class imbalance (-classWeighting)
  - It counts the number of instances for each category and use the normalized reverse frequency to scale learning rates per category.
  
2. Emprically verified way to freeze convolutional layers of the network.
  - I tried everything suggested to freeze a pretrained network, however, I saw that any method still updates the model. In the end, I modified [nnlr](https://github.com/gpleiss/nnlr) in order to freeze the network without any such leak. nnlr is a library that you can scale learning rates per layer. I changed the code to give a exact value per layer instead of scaling the base learning rate. The idea is to give 0 learning rate and weight decays to each of feature layers and prevent the model updating parameters.
  
3. Better booking of the trained models. 
  - Any model trained is arraged in a folder named by the important model parameters and sub-foldered by the date of the execution.
  
4. Plotting accuracy and loss values 
  - In the created folder for training model, there are loss and accuracy plots using [gnuplot](https://github.com/torch/gnuplot), plotting per epoch values.
  
5. New models;
  - GoogleNet
  - ResNet with Stochastic Depth
  - SimpleNet (a small architecture which is a good baseline)
  - And some others
  
6. Model initialization with a different learning rate (-model_init_LR)
  - It is good to stabilize a model before setting the learning rate to a base value. Given value is used for initial 5 epochs.
  
7. Save the model optimState so that you can continue the training from any checkpoint with all history recovered.

8. dataset/balanced.lua for balancing instance selection against imabalnced datasets

9. Set optimizer adam or sgd (-optimizer (sgd))


<b>NOTE</b>: Check other branches of the project. Eacn includes a particular model architecture.

- <b>Siamese:</b> Learning embeddings of data based on instance similarity.http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
- <b>TripletNet:</b> Learning embeddings of data based on instance similarity. https://arxiv.org/pdf/1412.6622.pdf
- <b>Regeression:</b> It is the same network structure but the code is tuned for Regression.

WARNING: " I suggest you to use this repo with caution since codes are only used for research purposes and there might be buggy details."
