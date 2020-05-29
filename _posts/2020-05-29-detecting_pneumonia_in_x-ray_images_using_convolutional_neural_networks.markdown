---
layout: post
title:      "Detecting Pneumonia in X-Ray Images Using Convolutional Neural Networks"
date:       2020-05-29 08:18:18 +0000
permalink:  detecting_pneumonia_in_x-ray_images_using_convolutional_neural_networks
---


Correctly diagnosing conditions and diseases of patients quickly and accurately allows for the timely treatment of patients as well as freeing up time for medical professionals that would have otherwise been spent diagnosing. If we can create a deep learning model that can accurately classify whether a patient has a condition or not, medical professionals will be able to use the models to better diagnose their patients. 

I'll be attempting to create a deep learning model that can accurately classify whether or not a patient has pneumonia or not by looking at their x-rays. I'll be making 4 different models:

1. Baseline Neural Network
2. Convolutional Neural Network
3. CNN with added layers and regularization
4. Transfer Learning with VGG16


Lets begin by taking a look at some x-ray images of the dataset. The dataset can be found [here](http://https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Normal
![](https://i.imgur.com/XiAsJFS.png)

### Pneumonia
![](https://i.imgur.com/DCoj0cd.png)

From a quick glance, normal lungs seem to look more clear than lungs with pneumonia but the difference isn't very clear. We'll see if the neural networks I create can tell the difference between the two.

## Data Preparation

The dataset comes with the data already split between training, testing, and validation sets. I'll begin by creating variables for the folder locations and resizing the images so that they are all the same size, I used image dimensions of 120x120. 
```
train = './chest_xray/train'
test = './chest_xray/test'
val = './chest_xray/val'

train_generator = ImageDataGenerator(rescale=1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2).flow_from_directory(
                    train,
                    target_size=(120, 120),
                    batch_size=5216,
                    class_mode='binary')

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                    test,
                    target_size=(120, 120),
                    batch_size=624,
                    class_mode='binary')

val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                    val,
                    target_size=(120, 120),
                    batch_size=16,
                    class_mode='binary')
```

Since I'll be starting by making a baseline neural network without any convolutional layers, I'll need to reshape all the images so that they have one input feature vector.

```
train_images, train_labels = next(train_generator)
test_images, test_labels = next(test_generator)
val_images, val_labels = next(val_generator)

train_img = train_images.reshape(train_images.shape[0], -1)
test_img = test_images.reshape(test_images.shape[0], -1)
val_img = val_images.reshape(val_images.shape[0], -1)

train_y = np.reshape(train_labels, (5216,1))
test_y = np.reshape(test_labels, (624,1))
val_y = np.reshape(val_labels, (16,1))
```

## 1. Baseline Neural Network

Now that we've got all our data set up, lets move on to making our baseline model.
```
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(43200, )))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

histoire = model.fit(train_img,
                     train_y,
                     epochs=50,
                     batch_size=64,
                     validation_data=(val_img, val_y))
```

After running this model we get a training accuracy of 94% and a testing accuracy of 81%. A testing accuracy of 81% is not great for the application of this model. Lets see if we can improve our performance using convolutional neural networks.

## 2. Convolutional Neural Network
Convolutional neural networks are the most popular neural network used for image classification. They perform a better fitting to the image dataset due to the reduction in the number of parameters involved and also the reusability of weights. Before we start with the model we'll need to remake our train generator with a smaller batch size because we'll be using fit_generator to run the models.
```
train_generator = ImageDataGenerator(rescale=1./255, 
                                     shear_range=0.2,
                                     zoom_range=0.2).flow_from_directory(
                    train,
                    target_size=(120, 120),
                    batch_size=32,
                    class_mode='binary')
```

Now that we've gotten that taken care of we can move on to creating our model.
```
cnn1 = models.Sequential()
cnn1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)))
cnn1.add(layers.MaxPooling2D((2, 2)))

cnn1.add(layers.Conv2D(32, (3, 3), activation='relu'))
cnn1.add(layers.MaxPooling2D((2, 2)))

cnn1.add(layers.Flatten())
cnn1.add(layers.Dense(128, activation='relu'))
cnn1.add(layers.Dense(1, activation='sigmoid'))

cnn1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
							
history1 = cnn1.fit_generator(train_generator, 
                              steps_per_epoch = 163,
                              epochs = 20,
                              validation_data = val_generator)
```

For the first model, I used 2 convolution layers with a hidden layer of 128 units and finally our output layer. The steps per epoch should be the size of the dataset divided by batch size, so in this case we have 5216 training examples divided by 32 which equals 163.

After the model has run, I'll calculate the evaluation metrics using this block of code
```
preds = cnn1.predict(test_images)

acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print(f'Accuracy: {acc}%')
print(f'Precision: {precision}%')
print(f'Recall: {recall}%')
print(f'F1-score: {2*precision*recall/(precision+recall)}%')
print(f'Specificity: {tn/(tn + fp)}%')

print('\nTRAIN METRIC ----------------------')
print(f'Train acc: {np.round((history1.history["acc"][-1])*100, 2)}%')
```
```
CONFUSION MATRIX ------------------
[[154  80]
 [  2 388]]

TEST METRICS ----------------------
Accuracy: 86.85897435897436%
Precision: 82.90598290598291%
Recall: 99.48717948717949%
F1-score: 90.44289044289043%
Specificity: 0.6581196581196581%

TRAIN METRIC ----------------------
Train acc: 96.97%
```

The CNN performs better than a baseline neural network with a 6% increase in accuracy over the baseline model.

We want to maximize our recall and specificity score for this problem with more weight placed on recall. 
> Recall is the true positive weight (Of all the people that have pneumonia, recall is the percentage of people the model correctly predicted having pneumonia)

>  Specificity is the true negative weight (Of all the people that do not have pneumonia, specificity is the percentage of people the model correctly predicted not having pneumonia)

Maximizing these two metrics is important because we want to be confident in our models ability to correctly identify whether or not a patient has pneumonia. I place more emphasis on recall because a false negative is more detrimental than a false positive in disease diagnosis. Minimizing false negatives is important for this problem because we don't want to misdiagnose a patient with pneumonia. The recall for this model is extremely high at 99% however the specificity is low at 66% due to the high number of false positives.

Next we'll see if adding more layers to our CNN improves model performance.

## 3. CNN with Added Layers and Regularization
I'll be adding multiple convolutional and hidden layers for this model. I also use dropout regularization in this model to combat overfitting.
```
cnn2 = models.Sequential()
cnn2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)))
cnn2.add(layers.Conv2D(32, (3, 3), activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))

cnn2.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn2.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))

cnn2.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn2.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))

cnn2.add(layers.Flatten())
cnn2.add(layers.Dense(1024, activation='relu'))
cnn2.add(layers.Dropout(0.3))
cnn2.add(layers.Dense(512, activation='relu'))
cnn2.add(layers.Dropout(0.3))
cnn2.add(layers.Dense(128, activation='relu'))
cnn2.add(layers.Dropout(0.3))
cnn2.add(layers.Dense(1, activation='sigmoid'))

cnn2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
							
history2 = cnn2.fit_generator(train_generator, 
                              steps_per_epoch = 163,
                              epochs = 20,
                              validation_data = val_generator)
```
```
CONFUSION MATRIX ------------------
[[199  35]
 [  6 384]]

TEST METRICS ----------------------
Accuracy: 93.42948717948718%
Precision: 91.64677804295943%
Recall: 98.46153846153847%
F1-score: 94.93201483312731%
Specificity: 85.04273504273505%

TRAIN METRIC ----------------------
Train acc: 96.49%

```

Adding extra layers and using dropout regularization increased accuracy by 7% from the previous CNN model. This model performs extremely well with an accuracy of 93%. Recall dropped by 1% however specificity increased from 66% to 85% showing a large improvement over our previous model.

## 4. Transfer Learning with VGG16
Transfer learning uses a pretrained network that has been trained on a large dataset and allows us to utilize the knowledge from the pretrained network and apply it to our data. 

We'll be using the VGG16 model which has been pretrained on over a million images from the ImageNet database. I'll be freezing the first 3 convolution layers so that the weights don't change for those, and I'll unfreeze the last 2 blocks so that their weights get updated as I pass new data into the model.

```
input_tensor = layers.Input(shape=(120, 120, 3))
vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

output = vgg16.layers[-1].output
output = layers.Flatten()(output)
vgg16 = models.Model(vgg16.input, output)

set_trainable=False
for layer in vgg16.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

pd.set_option('max_colwidth', -1)
layers1 = [(layer, layer.name, layer.trainable) for layer in vgg16_trainable.layers]
pd.DataFrame(layers1, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
```
![](https://i.imgur.com/jw8wxc8.png)

The last 2 convolution blocks are now trainable so we can continue on building our model.

```
model_vgg_train = models.Sequential()
model_vgg_train.add(vgg16)
model_vgg_train.add(layers.Dense(512, activation='relu', input_dim=vgg16_trainable.output_shape[1]))
model_vgg_train.add(layers.Dropout(0.3))
model_vgg_train.add(layers.Dense(128, activation='relu'))
model_vgg_train.add(layers.Dropout(0.3))
model_vgg_train.add(layers.Dense(1, activation='sigmoid'))

model_vgg_train.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
							
history4 = model_vgg_train.fit_generator(train_generator,
                               steps_per_epoch=163,
                               epochs=40,
                               validation_data=val_generator)
```
```
CONFUSION MATRIX ------------------
[[177  57]
 [  2 388]]

TEST METRICS ----------------------
Accuracy: 90.5448717948718%
Precision: 87.19101123595505%
Recall: 99.48717948717949%
F1-score: 92.93413173652692%
Specificity: 75.64102564102564%

TRAIN METRIC ----------------------
Train acc: 98.16%
```

Utilizing transfer learning with the VGG16 model gives us an accuracy score of 90%, a recall of 99%, and a specificity of 75%. Although the recall score increased with this model, the accuracy and specificity are lower than our previous model.

## Conclusion

After testing multiple models with our data, the one that works best for this problem was our CNN model with extra layers. With an accuracy score of **93%**, recall of **98%**, and specificity of **85%** I would be very confident in using this model to detect pneumonia from x-ray images. I hope this gave you an insight into building neural networks for image classification.
