import cython_util
import numpy as np


all_results = [[np.array([[10,20,30,40,0.6]])]]
inp_data = [np.ones((512,512,3))]
offset = 2
output = cython_util.crop_images(all_results, inp_data, offset)
print(output)