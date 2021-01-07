import function as rps
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model=load_model('C:/Users/ADMIN/Desktop/python/rock paper scissors/my_model.h5')

rock_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/rock')
paper_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/paper')
scissor_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/scissors')
data=np.vstack((np.asarray(rock_dataset),np.asarray(paper_dataset),np.asarray(scissor_dataset)))
input=np.asarray(data[1600])
outcome=rps.predict(model,input.reshape(1,150,150,3))

label=''
if outcome[0]==0:
    label='it\'s a rock'
if outcome[0] == 1:
    label = 'it\'s a paper'
if outcome[0] == 2:
    label = 'it\'s a scissor'
plt.imshow(input)
plt.xlabel(label,color='red')
plt.show()

