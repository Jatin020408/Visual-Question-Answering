# Visual-Question-Answering
The project is  aimed to help the visually impaired by giving them the ability to take a picture, ask questions about it and the application will provide them with the answers using machine learning techniques and tools.


## VQA model:
### Dataset used:
For our task, we used the <a href=https://visualqa.org/download.html> VQA dataset </a>, it contains 0.25M images, 0.76M questions, and 10M answers.

### Data Pre-processing:
The preprocessing phase can be split into three main procedures:  
**1. Creating the questions vocabulary:**  
We created a vocabulary that contains all the words that appeared in the questions training set, we finally added special tokens for <pad> and <unk> to that vocabulary.  
**2. Creating the answers vocabulary:**  
Similar to the question vocabulary, this answers vocabulary contains the top 1000 most frequent answers that appeared in the training set.  
**3. Image preprocessing and feature extraction:**  
We first resized the images into (224, 244) to be compatible with the VGG-16 input layer, then the images were converted from RGB to BGR, and each color channel is zero-centered.
We then used a pretrained VGG-16 model to extract the features from these pre-processed images and stored them to be later passed to our model.

### Model Architecture:
1. **Image encoder:**   
It takes the pre-processed image features extracted using VGG-16 convolution neural network. These features were stored with the shape of (49, 512). The model flattens the image features and then feeds them to a fully connected layer with 1024 neurons and uses the RELU activation function. This part outputs a 1024-dim embedding of the image.  
2. **Question encoder:**  
It takes a padded tensor of the vocabulary indices for each word in the question sentence. This tensor has the length of max_qu_length = 30. It uses an embedding layer initialized using pre-trained GloVe-300 word embeddings to replace each word in the sentence with its representative 300-dim vector. The word embeddings are then fed into a recurrent neural network with LSTM cells that has a hidden state dimension of 1024. The hidden state of the last LSTM cell provides a 1024-dim embedding of the question.  
3. **Answer predictor:**  
In this part, the image embedding and the question embedding are fused together using element-wise multiplication. The results of the multiplication are then fed into a drop-layer with a drop rate of 20%, which helps to prevent overfitting, and then into a fully connected layer of K = 1000 neurons and it uses softmax activation function to provide a probability distribution over K answers
![image](https://user-images.githubusercontent.com/44211916/170712395-182387b4-5013-417c-9c3a-bd63a26cab92.png)

### Model Training:
All of the model training was done on the cloud using Google Colab.  
The model the GloVe word embedding representation for each word in the question, which was then passed to the RNN, to extract the questions' features representation and the already extracted image features, fused them together, and passed them to MLP, which had a softmax final layer that gave the probability distribution over the 1000 answers in our set.  

