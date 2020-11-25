# Spoken Language Identification
> homework for EE 599 - Deep Learning System, Fall 2020, USC
## Intro
In the spoken language identification task of classifying 3 different languages (English, Hindi, Mandarin), I firstly extract 64 features using MFCC from audio, then a GRU based model is used on the deep learning construction.

## Training Performance
For accuracy of training and validation set, they tend to perform well. At the last epochs, accuracy for training set comes to 97.3%, accuracy for validation set comes to 94.1%.
We could say accuracy performance is pretty good, with no sign of overfitting or underfitting.


## Testing Performance
For testing, I change the output (batch_size, len_sequence, num_features), so that the model can predict every single frame. So, output will be (1, 6000, 3), with 3 softmax results.

I separate whole training and testing dataset with portion of 90:10. The accuracy of testing dataset is 63.8%. The accuracy is lower because none of the audio file are trained, showing that generation performance of our model is not good though. One important reason is that the loss function is still not trained rightly even if I set weight for imbalanced data.

