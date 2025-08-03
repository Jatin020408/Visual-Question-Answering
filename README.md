
#  Visual Question Answering (VQA) System

This project implements a deep learning-based Visual Question Answering (VQA) model that takes an image and a natural language question as input and predicts the most probable answer. It combines **Computer Vision** and **Natural Language Processing** to learn multi-modal interactions using an attention mechanism.

---

##  Architecture

![Architecture Diagram](A_README_graphic_displays_details_of_a_Visual_Ques.png)

---

##  Project Structure

- **Data Handling**
  - Dataset is loaded from `.npy` files containing images and questions.
  - Tokenization and indexing of both questions and answers using a vocabulary file.

- **Model Components**
  - `ImgAttentionEncoder`: Extracts spatial image features using pretrained VGG19.
  - `QstEncoder`: Encodes the question using LSTM after word embeddings.
  - `Attention`: Applies attention over image regions based on the question context.
  - `SANModel`: Stacked Attention Network (SAN) that combines question and attended image features to make predictions.

- **Training Loop**
  - Uses `CrossEntropyLoss`, Adam optimizer, and step LR scheduler.
  - Early stopping based on validation loss.

- **Evaluation**
  - Top-5 predictions printed with confidence scores.
  - Also includes a CLI for testing single (image, question) inputs.

---

##  Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- PIL

Install required libraries:

```bash
pip install torch torchvision opencv-python numpy pillow
```

---

##  Directory Structure

```
/vqa
  ├── dataset/
  │   ├── train.npy
  │   ├── valid.npy
  │   ├── vocab_questions.txt
  │   ├── vocab_answers.txt
  │   └── Resized_Images/
  ├── models/
  ├── logs/
  ├── test.py
  ├── train.py
```

---




## 🧠 Example Output

```
Input Question: What does the sign say?
Predicted Answers:
'stop' - 0.9231  
'go' - 0.0324  
'yield' - 0.0223  
'turn' - 0.0115  
'one way' - 0.0091  
```

---

##  Evaluation Metric

Accuracy is calculated based on how many of the multiple-choice ground truths the model correctly predicts (Top-1).

---

## 📌 Key Hyperparameters

| Parameter         | Value  |
|-------------------|--------|
| max_qst_length    | 30     |
| embed_size        | 1024   |
| word_embed_size   | 300    |
| hidden_size       | 64     |
| num_layers (LSTM) | 2      |
| batch_size        | 256    |
| num_epochs        | 30     |
| learning_rate     | 0.001  |

---

## 🙌 Acknowledgements

- [VQA v2.0 Dataset](https://visualqa.org/)
- [PyTorch Documentation](https://pytorch.org/)
- Pretrained [VGG19](https://pytorch.org/vision/stable/models.html)
