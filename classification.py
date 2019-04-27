
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing import image


# In[2]:


image_x, image_y = 224,224
classifier = load_model('final_model.h5')


# In[3]:


test_image = image.load_img('test_images/zebra2.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image/=255.
result = classifier.predict(test_image)
if result[0][0] == 1:
    print('tiger')
elif result[0][1] == 1:
    print('zebra')
        
        


plt.imshow(test_image[0])
from keras import models
# Extracts the outputs of the top 8 layers: 
layer_outputs = [layer.output for layer in classifier.layers[:8]] 
# Creates a model that will return these outputs, given the model input: 
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
activations = activation_model.predict(test_image)

# intermediate layers output

layer_names = []
for layer in classifier.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16
i = 0

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
            
        
    plt.savefig('capture/'+ str(i) + layer_name+str(col)+'.png')
    i = i + 1
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

# In[8]:


# img_name = "testing_images/tiger4.jpg"
# save_img = cv2.resize(1, (image_x, image_y))
# #cv2.imwrite(img_name, save_img)
# #print("{} written!".format(img_name))
# img_text = predictor()


# In[9]:


# In[ ]:
# accuracy graph and loss graph
'''
plt.plot(classifier.history['acc'])
plt.plot(classifier.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.figsave('acc.jpg')

plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.figsave('loss.jpg')
'''
