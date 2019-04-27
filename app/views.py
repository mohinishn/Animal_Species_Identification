from django.shortcuts import render,redirect
from django.http import HttpResponse 
from app.forms import TrafficForm
from app.models import Traffic
import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing import image

def index(request): 
	if request.method == 'POST': 
		form = TrafficForm(request.POST, request.FILES) 
		if form.is_valid(): 
			form.save()
			obj = Traffic.objects.all()
			lastImageObj = obj[len(obj)-1]
			lastImageUrl = lastImageObj.Image_URL.url
			print(lastImageUrl)

			image_x, image_y = 224,224
			print("11111111111111111111111111111111111")
			classifier = load_model('final_model7.h5')
			print("22222222222222222222222222222222222")

			test_image = image.load_img('.'+lastImageUrl, target_size=(224, 224))
			test_image = image.img_to_array(test_image)
			test_image = np.expand_dims(test_image, axis = 0)
			test_image/=255.


			def predictor(test_image):
				result = classifier.predict(test_image)
				print(result)
				if result[0][0] >= 0.5:
					return 'Giraffe'
				elif result[0][1] >= 0.5:
					return 'Leopard'
				elif result[0][2] >= 0.5:
					return 'Lion'
				elif result[0][3] >= 0.5:
					return 'Panda'
				elif result[0][4] >= 0.5:
					return 'Rhinoceros'
				elif result[0][5] >= 0.5:
					return 'Tiger'
				elif result[0][6] >= 0.5:
					return 'Zebra'
				else:
					return 'Animal Not Found'

			result = predictor(test_image)
			print(result)

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


				plt.savefig('media/out/'+ str(i) + layer_name+'.png')
				print("media/out/"+str(i)+layer_name+".png")
				i = i + 1
				scale = 1. / size
				plt.figure(figsize=(scale * display_grid.shape[1],
									scale * display_grid.shape[0]))
				plt.title(layer_name)
				plt.grid(False)

				plt.imshow(display_grid, aspect='auto', cmap='viridis')
			return render(request,'app/index.html',{'result':result})

	else: 
		form = TrafficForm() 
	return render(request, 'app/index.html', {'form' : form}) 

def about(request):
	return render(request,'app/about.html')
