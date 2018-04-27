'''
1、获取mnist数据
2、把这些数据放到Inception中，获取mu， sigma
用所谓的官方方法遇到纬度不匹配问题，用pytorch实现
'''

import numpy as np
from inception import InceptionV3
from fid_score import calculate_activation_statistics
from util import load_mnist



data_name = 'MNIST'
# data_name = 'fashion-mnist'
images, labels = load_mnist(data_name)
images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2]))

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])

tmp = images.shape;

imgs_tmp = np.zeros([tmp[0], tmp[1], tmp[2], 3])
imgs_tmp[:, :, :, 0] = images
imgs_tmp[:, :, :, 1] = images
imgs_tmp[:, :, :, 2] = images
imgs_tmp = imgs_tmp.transpose((0, 3, 1, 2))

# 如果你没有有gpu的话，下面的参数请把True改为False
m, s = calculate_activation_statistics(imgs_tmp, model, 64, 2048, True)
np.savez('npy/' + data_name + '.npz', mu=m, sigma=s)
print('m.shape:' + str(m.shape))
print('s.shape:' + str(s.shape))
print(data_name + ' npy finished!!!')


















