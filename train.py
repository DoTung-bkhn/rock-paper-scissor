import function as rps,sys
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxsize)   #print full numpy array

#prepare data
rock_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/rock')
paper_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/paper')
scissor_dataset=rps.load_image('C:/Users/ADMIN\Desktop/python/rock paper scissors/data/scissors')
data=np.vstack((np.asarray(rock_dataset),np.asarray(paper_dataset),np.asarray(scissor_dataset)))

#label data
rock_num=len(rock_dataset);paper_num=len(paper_dataset);scissor_num=len(scissor_dataset)
target=rps.label(rock_num,paper_num,scissor_num)

#training model
model=rps.create_model()
data,target=shuffle(data,target)
history=model.fit(data,target,batch_size=32,epochs=30,validation_split=0.1)

rps.save_model(model)
