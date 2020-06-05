
  In this project, firstly, I loaded the dataset as object in python. The dataset contains three part: train data, validation data, test data. I have used sklearn functions to split the train data into train part and validation part, so the validation data is useless in my process.
  After loaded the data, I did the preprocess of these images. preprocess is the key to enhanec the result of the deep neural network. I have gray the color image and normalized them.
  In this project, I used LeNet-5 model as the classifier.
  Then, I designed my own architecture of network in referecnce of guihub. TIn my project, there are two convolution layers and three full connected layers. The kernal size is [5,5] in both convolution layers. The number in full connected layer of nodes is 64,256,256 in each layers.I use cross entropy to establish the loss function. The activation fucntion is relu. I have involved the dorpout methods into my project which can increase the accuracy of the classier.
  After 50 training steps, I got my result. The training accuracy is 100%, the accuracy in validation data is 98.8%. By the way, the accuracy in the test dataset is 93.6%. I have download some pictures from Internet, I use my model on these pitcures and the model can classify these pictures very well.
  


I used the pandas library to calculate summary statistics of the trafficsigns data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

My final model consisted of the following layers:
Input Layer: size:32*32*3
Convolutional Layer1: input size: 32*32*3, kernal size:5*5, kernal depth:32, output size:32*32*32; activation fucntion:Relu
Pooling Layer1: input size: 32*32*32 output size:16*16*32
Convolutional Layer2: input size:16*16*32, kernal size:5*5 kernal depth:64, output size: 16*16*64; activation function:Relu
Pooling Layer1:input size:16*16*64, output size:8*8*64
Full Connected Layer1: input size:16384*1, output size:256*1 activation function:Relu,dropout parameter:0.5
Full Connected Layer2: input size:256*1,  output size:256*1 activation function:Relu,dropout parameter:0.5
Full Connected Layer3: input size:256*1, output size:43*1 

The number of layers: two convolutional layers followed by three full connected layers. This is the same with classical model called LeNet-5.

I used cross entrocy as my loss function conbined with the softmax function.
In the final training steps, EPOCHS = 50S  BATCH_SIZE = 128
To train the model, I used tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit,labels = y),tf.reduce_mean(cross_entropy) and so on function to get a good output.


I am sorry that I do not record all tunning steps:
batch size: 256, epochs: 35, rate: 0.00075, mu: 0, sigma: 0.3 keep_prob: 0.5 training set accuracy of 100% validation set accuracy of 96.7% 
batch size: 64, epochs: 25, rate: 0.00075, mu: 0, sigma: 0.1 keep_prob: 0.5 training set accuracy of 93% validation set accuracy of 92.4% 
batch size: 100, epochs: 100, rate: 0.0009, mu: 0, sigma: 0.1 keep_prob: 0.5 training set accuracy of 100% validation set accuracy of 98.8% 
batch size: 100, epochs: 60, rate: 0.0009, mu: 0, sigma: 0.1, keep_prob: 0.5 training set accuracy of 100% validation set accuracy of 97.2% 
batch size: 128, epochs: 50, rate: 0.0009, mu: 0, sigma: 0.1, keep_prob: 0.5 training set accuracy of 100% validation set accuracy of 98.8% 

I decided to generate additional data because I want to validate my model in different data to make it more robustness.

To add more data to the the data set, I used the following techniques: glob fucntion and numpy. Glob help me to sort the new data.

I have downloaded five German traffic signs that I found the accuracy is 100% which is the same with the accuracy on the validation dataset.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.8% 
* test set accuracy of 93.6%






