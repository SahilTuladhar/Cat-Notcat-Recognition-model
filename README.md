# Cat vs Not-Cat Image Classifier

🐾 A Machine Learning Project for identifying images as either containing a cat or not, using neural networks created using logistic regression algorithm and image processing techniques.

## Description 🐱📸

This project focuses on building a neural network to classify images as either a "cat" or "not-cat." It uses image preprocessing, logistic regression, and possibly deeper neural networks for image classification. This model is trained using a labeled dataset of cat and non-cat images and evaluated for accuracy.

#### 1. Image Preprocessing:

- Images are resized, normalized, and flattened to create input vectors for the model.

#### 2. Neural Network Development:

- A simple feedforward neural network (or logistic regression) is implemented for binary classification which is not a deep neural network

#### 3. Model Development

- Built a feed forward neural network with the following architecture:

  - Input layer: Each image has dimensions (num*px, num_px, 3) (where 3 represents the RGB color channels), then the total number of input features is num_px * num*px * 3.
  - Single Logistic Regression Layer: - Logistic regression can be thought of as a single hidden layer - Weights (w): This is a vector of parameters with shape (num*px * num*px * 3, 1). Each pixel in the flattened image has an associated weight. These weights are initialized as zeros in the initialize_with_zeros(dim) function.
  - Bias (b): A scalar parameter that is initialized to zero. This bias allows the model to fit the data better by shifting the decision boundary.
  - Activation Function: The logistic regression model applies the sigmoid activation function to the weighted sum of the inputs. The function is σ(Z) where  
    𝑍=𝑤𝑇⋅𝑋+𝑏, This gives a probability value A between 0 and 1.

  - Output layer:
    - The output is a binary classification: it outputs 1 if the image is predicted to contain a cat and 0 otherwise.

#### 4.Model Training and Evaluation

- The model is trained using backpropagation where the concept of gradient descent is implemented
- Evaluated the model’s performance using test data and calculated the accuracy of predictions.

#### 5.Visualization

- Displayed sample images and their predicted labels to visualize model performance.

## Limitations ⛔️

- The model may struggle with images that contain noise or ambiguous content.
- Works best with well-lit and properly framed images.
- Incorrect classification of objects that are similar to cats. Eg. Dog
- As a simple single layer Neural network is implemented the model ovefits which is supported by the image below.

![overfitting](https://github.com/SahilTuladhar/Cat-Notcat-Recognition-model/blob/master/images/images/overfit.png)

## Code Requirements 📱

- Install Conda for Python or create a virtual environment for dependency management.
- To run this project, you need to install the following dependencies:

- NumPy
- Matplotlib
- h5py
- PIL
- datasets

```bash
  pip install th5py numpy matplotlib PIL datasets
```

## Execution ▶️

Run the notebook or Python scripts in the following sequence:

1. Data preprocessing
2. Model training
3. Evaluation and visualization

## Results 📈

- Correct classification as a cat

  ![isacat](https://github.com/SahilTuladhar/Cat-Notcat-Recognition-model/blob/master/images/images/is_a_cat.png)

- Correct classification as car is not a cat

  ![isnotacat](https://github.com/SahilTuladhar/Cat-Notcat-Recognition-model/blob/master/images/images/is_not_a_cat.png)

- Misclassification as some images of dog may have similar properties to a cat

  ![misclassify](https://github.com/SahilTuladhar/Cat-Notcat-Recognition-model/blob/master/images/images/mis_classify.png)

- Decreasing loss function as gradient descent is applied

![loss](https://github.com/SahilTuladhar/Cat-Notcat-Recognition-model/blob/master/images/images/loss%20decrease.png)
