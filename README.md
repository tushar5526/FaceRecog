# FaceRecog
## Face Recogntion project based on Transfer learning using VGG 16 for MLops task

### Project Structure 

***faceData***

 - **train**
   - n1
     - (training images for n1)
   - n2
     - (training images for n2)
     
 - **test**
   - n1
     - (test images for n1)
   - n2
     - (test images for n2)
   
 
You can collect and label your own data by using ***click_your_photo.py***, this script collects 100 photos of your by default and saves it in a folder named FaceData, the images names are by default stored as ***1_(image_count).jpg***
 
First project of mine with tensorflow on Docker container, it is a face recoginition model build using keras on TF backend
The results can be tested using Pillow as I am having issues with running OpenCV on container, but will get on it in future 


===========================================================================================================================

# Import :

Import the required modules 

```
from keras.applications import vgg16
import keras

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D

from keras.optimizers import RMSprop

from keras.models import load_model
from PIL import Image
import numpy as np
```

# Initial Model :

```
model = vgg16.VGG16(weights='imagenet',include_top = False)
```
***Exclude the top most softmax layer and set layers to trainable***

```
for l in model.layers:
 l.trainable = False
```

***Our FC Head or model***

*VGG -> AveragePool -> Dense(relu) -> Dense(relu) -> Dense(softmax) -> Result*
```
top_model = model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(1024,activation='relu')(top_model)
top_model = Dense(1024,activation='relu')(top_model)
top_model = Dense(2,activation='softmax')(top_model)

```

### New Model = VGG16 + FC HEAD 

```
nmodel = Model(inputs = model.input, outputs = top_model)
```

# Get train and test Data 

```
img_rows, img_cols = 224,224
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'faceData/train/'
validation_data_dir = 'faceData/test/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='categorical')
```


# Train the model 

```
from keras.optimizers import RMSprop

nmodel.compile(loss = 'categorical_crossentropy'
              ,optimizer = RMSprop(lr = 0.001), metrics = ['accuracy'])


#Enter the number of training and validation samples here

# We only train 5 EPOCHS 
epochs = 5

history = nmodel.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size)

```

# Test Model on your data :

```
from keras.models import load_model
from PIL import Image
import numpy as np
classifier = load_model('faceRecog.h5')
input_im = Image.open("faceData/test_tushar.jpg")
input_im.show()
input_original = input_im.copy()

input_im = input_im.resize((224, 224))
display(input_im)
input_im = np.array(input_im)
input_im = input_im / 255.
input_im = input_im.reshape(1,224,224,3)
res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

if res == [0]:
    print('n1 : NAME')
elif res == 1:
    print('n2 : NAME')

```

