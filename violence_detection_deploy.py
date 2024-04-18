#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
#from PIL import Image


# In[2]:


model = load_model('my_model.h5')


# In[ ]:





# In[5]:


# Load and preprocess the image you want to test
#test_image_path = 'test/pose.jpg'
#test_image = preprocess_image(test_image_path)

test_image = np.load('test/idle.npy')
test_image = np.load('test/highfive.npy')
test_image = np.load('test/pose.npy')
test_image = np.load('test/slap.npy')
test_image = np.load('test/grapple.npy')
test_image = np.load('test/club.npy')
test_image = np.load('test/kick.npy')


# In[6]:


# Make a prediction
prediction = model.predict(test_image)

classes = ['idle','walk','highfive','pose','slap','grapple','club','kick','pose']

# Decode the prediction
predicted_class_index = np.argmax(prediction, axis=1)
#predicted_class = label_encoder.inverse_transform(predicted_class_index)  # Ensure label_encoder is the one used during training

print(f"The model predicts: {classes[predicted_class_index[0]]}")


# In[ ]:





# In[ ]:




