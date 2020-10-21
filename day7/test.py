import torch
import numpy as np


# 测试TorchVision Object Detection Finetuning Tutorial中关于mask的操作
# mask是二维的,标记图片中人位置和id
def test0():
    masks = torch.zeros(2,4,4, dtype=float)
    masks[0, :, 1] = 1
    masks[0, :, 2] = 2
    masks[1, :, 3] = 3
    # print(masks)
    '''
    tensor([[[0., 1., 2., 0.],
         [0., 1., 2., 0.],
         [0., 1., 2., 0.],
         [0., 1., 2., 0.]],
         
        [[0., 0., 0., 3.],
         [0., 0., 0., 3.],
         [0., 0., 0., 3.],
         [0., 0., 0., 3.]]], dtype=torch.float64)
          '''

    mask = masks[0]

    # 测试numpy.unique()
    test_data = [1, 3, 4, 4]
    # print(np.unique(test_data)) # [1 3 4]
    obj_ids = np.unique(mask)
    # print(obj_ids) # [0. 1. 2.]
    # print(obj_ids.shape) # (3,)
    test_data = np.array([1, 2, 3])
    # print(test_data) # [1 2 3]

    obj_ids = obj_ids[1:]
    # print(type(obj_ids)) # numpy.ndarray
    mask = mask.numpy()
    # print(mask.shape) # (3, 4, 4)
    # mask = mask.numpy()

    # obj_ids = obj_ids[:, None, None]
    # print(obj_ids)
    '''
    [[[1.]]
    [[2.]]]
    '''
    # print(obj_ids.shape) # (2, 1, 1)


    # mask是二维的！！！
    # print(mask)
    '''
    [[0. 1. 2. 0.]
    [0. 1. 2. 0.]
    [0. 1. 2. 0.]
    [0. 1. 2. 0.]]
    '''
    masks = mask == obj_ids[:, None, None]
    # print(masks)
    '''
    [[[False  True False False]
  [False  True False False]
  [False  True False False]
  [False  True False False]]
 [[False False  True False]
  [False False  True False]
  [False False  True False]
  [False False  True False]]]  (2,4,4) 得到了两个目标的位置
    '''

    '''
    masks = mask == obj_ids[:, None, None]
    这个操作使用了广播
    mask维度为 4 * 4
    obj_ids[:, None, None] 维度为 (2, 1, 1)
    obj_ids = obj_ids[:, None, None]
    obj_ids[1] : [[1]]
    '''

if __name__ == '__main__':
    test0()
