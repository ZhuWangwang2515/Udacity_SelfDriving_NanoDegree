

  I uesd the te
  In this project, firstly, I need to load the dataset as object in python. The dataset contains three part: train data, validation data, test data. I have used sklearn functions to split the train data into train part and validation part, so the validation data is useless in my process.
  After loaded the data, I did the preprocess of these images. preprocess is the key to enhanec the result of the deep neural network. I have gray the color image and normalized them.
  Then, I designed my own architecture of network in referecnce of guihub. In my project, there are two convolution layers and three full connected layers. The kernal size is [5,5] in both convolution layers. The number in full connected layer of nodes is 64,256,256 in each layers.I use cross entropy to establish the loss function. The activation fucntion is relu. I have involved the dorpout methods into my project which can increase the accuracy of the classier.
  After 50 training steps, I got my result. The training accuracy is 100%, the accuracy in validation data is 98.8%. By the way, the accuracy in the test dataset is 93.6%. I have download some pictures from Internet, I use my model on these pitcures and the model can classify these pictures very well.
  Thanks for your reading!
  -

















