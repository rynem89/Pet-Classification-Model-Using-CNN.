"""
This code implements a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras.
The code performs the following steps:
1. Imports the required libraries.
2. Sets the paths for training and testing data.
3. Counts the number of training samples for each category.
4. Sets the paths for dog and cat images.
5. Removes existing preview images.
6. Defines the image data generator for data augmentation.
7. Augments the training images and saves the preview images.
8. Creates a DataFrame for the training images.
9. Displays a few images from the training dataset.
10. Creates a Normalising generator for the training images.
11. Creates a DataFrame for the testing images.
12. Displays all images from the testing dataset.
13. Creates a Normalising generator for the testing images.
14. Defines the CNN model architecture.
15. Compiles the model.
16. Trains the model on the training data.
17. Saves the trained model.
18. Evaluates the model on the testing data.
19. Displays the predicted labels for the testing images.
20. Plots the validation loss and accuracy for different epochs.
"""
# Import required libraries

# Import the TensorFlow library
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
import os.path
import matplotlib.pylab as plt
import glob
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from keras.utils import img_to_array, load_img

# Set the paths for training and testing data
train_dir = 'data/train'
test_dir = 'data/test'

# Get the number of training samples for each category
dogs = len(os.listdir(os.path.join(train_dir, 'dogs')))
cats = len(os.listdir(os.path.join(train_dir, 'cats')))
print('Number of training samples for Dog category =', dogs)
print('Number of training samples for Cat category =', cats)

# Set the paths for dog and cat images
dog_dir = glob.glob('data/train/dogs/*.jpg')
cat_dir = glob.glob('data/train/cats/*.jpg')

# Set the paths for preview images
cat_dir_p = glob.glob('train/Cat_preview/*.jpeg')
dog_dir_p = glob.glob('train/Dog_preview/*.jpeg')

# Remove existing preview images
for d_dir, c_dir in zip(dog_dir_p, cat_dir_p):
    try:
        os.remove(d_dir)
        os.remove(c_dir)
    except:
        print('nothing to delete')

# Define the image data generator for data augmentation
aug_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')

# Augment the training images and save the preview images
j=0
i=0
for j in range(2):
    if j == 0:
        dir= cat_dir
        save_dir = 'train\Cat_preview'
        prefix='cat'
    else:
        dir= dog_dir
        save_dir = 'train\Dog_preview'
        prefix='dog'
    for loc_img in dir:
        img = load_img(loc_img)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (x, x, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, x, x, 3)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in aug_datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=prefix, save_format='jpeg'):
            if i > 50:
                break  # otherwise the generator would loop indefinitely
            i = i +1
    j=j+1

# Create a DataFrame for the training images
# List to store image paths
image_paths = []
# List to store corresponding labels
labels = []
i=0
for i in range(2):
    # Set the image directory path (adjust to your actual path)
    if i == 0:
        image_dir = r"train\Cat_preview"
        start = 'cat'
    else:
        image_dir = r"train\Dog_preview"
        start = 'dog'
    # Loop through all files in the image directory and subdirectories
    for filename in os.listdir(image_dir):
        # Get the full path to the file
        file_path = os.path.join(image_dir, filename)
        # Check if it's a JPG file
        if filename.startswith(start):
            # Add the path to the image paths list
            image_paths.append(file_path)
            # Extract the label based on the directory name (adjust if needed)
            label = os.path.basename(os.path.dirname(file_path))
            if label == 'Cat_preview':
                label = 'Cat'
            elif label == 'Dog_preview':
                label = 'Dog'
            # Add the label to the labels list
            labels.append(label)
# Create pandas Series for paths and labels
filepaths_series = pd.Series(image_paths, name="Filepath")
labels_series = pd.Series(labels, name="Label")
# Create the DataFrame by combining the Series
train_df = pd.concat([filepaths_series, labels_series], axis=1)
print(train_df)

# Display a few images from the training dataset
stored_img = np.empty(len(train_df), dtype=object)
i=0
plt.figure(figsize=(10,7))
plt.title('Few Images of training dataset', y=0.1)
x=np.random.randint(0, len(train_df))
for training_img_loc in train_df.loc[988:1020,'Filepath']:
    stored_img[i] = plt.imread(training_img_loc)
    ax = plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.imshow(np.array(stored_img[i]).astype("uint8"))
    ax.set_title(train_df.loc[988+i,'Label'])
    plt.axis("off")
    i = i+ 1
    if i == 20:
        break
plt.tight_layout()
plt.show()
train_df = train_df.sample(frac = 1) # Shuffle the DataFrame
# Print the DataFrame
print(train_df)

# Create a generator for Normalising the training images
train_generator = ImageDataGenerator(rescale = 1./255)
# Transfroming the image data and its label to a list numpy arrays
train_images = train_generator.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = (256,256),
    classes=['Cat','Dog'],
    class_mode = 'categorical',
    batch_size = 64,
    color_mode = 'rgb',
    seed = 42,
    shuffle = True,
    subset = 'training')

# Create a DataFrame for the testing images
image_paths = []
labels = []
for i in range(2):
    if i == 0:
        image_dir = r"data\test\cats"
    else:
        image_dir = r"data\test\dogs"
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if filename.endswith(".jpg"):
            image_paths.append(file_path)
            label = os.path.basename(os.path.dirname(file_path))
            if label == 'cats':
                label = 'Cat'
            elif label == 'dogs':
                label = 'Dog'
            labels.append(label)
filepaths_series = pd.Series(image_paths, name="Filepath")
labels_series = pd.Series(labels, name="Label")
test_df = pd.concat([filepaths_series, labels_series], axis=1)
# Print the DataFrame
print(test_df)

# Display all images from the testing dataset
stored_img = np.empty(len(train_df), dtype=object)
i=0
plt.figure(figsize=(10,7))
plt.title('All Images of Validation/Testing dataset', y=0.1)
for training_img_loc in test_df.loc[:,'Filepath']:
    stored_img[i] = plt.imread(training_img_loc)
    ax = plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.imshow(np.array(stored_img[i]).astype("uint8"))
    ax.set_title(test_df.loc[i,'Label'])
    plt.axis("off")
    i += 1
    if i == len(test_df):
        break
plt.tight_layout()
plt.show()

# Create a generator for Normalising the testing images
test_generator = ImageDataGenerator(rescale = 1./255)
# Transfroming the image data and its label to a list numpy arrays 
test_images = test_generator.flow_from_dataframe(
    dataframe = test_df,
    x_col = 'Filepath',
    y_col = 'Label',
    target_size = (256, 256),
    classes=['Cat','Dog'],
    class_mode = 'categorical',
    color_mode = 'rgb',
    batch_size = 20,
    seed = 42,
    shuffle = True)

# Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same',input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax')])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save_weights('model.h5')
model.save_weights('best_model.h5')
model.summary()

j=0
for j in range(3):
    j=j+1
    model.load_weights('model.h5') # load original weights for consistency for different epochs
    history = model.fit(
    train_images,
    validation_data=test_images,
    epochs=100*j,
    callbacks=[   # callback to save best accuracy model weights and reduce learning rate if loss remains same
        tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5",monitor="val_accuracy",verbose=1,save_best_only=True,save_weights_only=True,
        mode="auto",save_freq="epoch",initial_value_threshold=None,),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=20,min_lr=0.0001,factor=0.1)])

    # Save the model
    model.save('my_model')

    model.load_weights('best_model.h5')
    # Evaluate the model on the test data
    results = model.evaluate(test_images, verbose = 0)
    print('Test Loss: {:.5f}'.format(results[0]))
    print('Test Accuracy: {:.2f}%'.format(results[1] * 100))

    # Display the predicted labels for the testing images for 100, 200, and 300 epochs
    test = np.empty(len(train_df), dtype=object)
    i=0
    plt.figure(figsize=(10,7))
    if j == 1:
        history_100 =history 
        plt.title('All Images Predicted labels for Validation/Testing dataset for 100 epochs',y= 0.1)
    elif j==2:
        history_200 =history
        plt.title('All Images Predicted labels for Validation/Testing dataset for 200 epochs',y= 0.1)
    else:
        history_300 =history 
        plt.title('All Images Predicted labels for Validation/Testing dataset for 300 epochs',y= 0.1)
    for img in test_df.loc[:,'Filepath']:
        test[i] = plt.imread(img)
        ax = plt.subplot(5, 5, i + 1)
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.imshow(np.array(test[i]).astype("uint8"))
        x = img_to_array(test[i])  
        x = cv.resize(x,(256,256))
        x = x.reshape((1,) + x.shape)
        predicted = model.predict(x)
        print(predicted)
        result= np.argmax(predicted, axis=-1)
        print(train_images.class_indices)
        if result[0]==0:
            prediction= 'Cat'
        else:
            prediction = 'Dog'
        ax.set_title(prediction)
        plt.axis("off")
        i += 1
        if i == len(test_df):
            break
    plt.tight_layout()
    plt.show()
    

plt.figure(figsize=(10,9))
plt.title('Plotting the Testing and Validation loss and accuracy for 100, 200 and 300 epochs',y=0.1)
j=0
for j in range(3):
    j=j+1
    if j == 1:
        history = history_100
        loss= '100 epoch loss'
        accuracy='100 epoch accuracy' 
    elif j==2:
        history = history_200
        loss= '200 epoch loss'
        accuracy='200 epoch accuracy' 
    elif j==3:
        history = history_300
        loss= '300 epoch loss'
        accuracy='300 epoch accuracy' 
# Plot the validation loss and accuracy for different epochs
    ax = plt.subplot(3,3,j)
    ax.plot(history.history["loss"],c = "purple")
    ax.plot(history.history["val_loss"],c = "orange")
    plt.title(loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["train", "test"])
    ax = plt.subplot(3,3,j+3)
    ax.plot(history.history["accuracy"],c = "purple")
    ax.plot(history.history["val_accuracy"],c = "orange")
    plt.title(accuracy)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["train", "test"])
plt.show()
