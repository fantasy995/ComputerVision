import torch
import torch.nn as nn
import numpy as np

batch_size = 100


# *args 表示任何多个无名参数，它本质是一个 tuple
# **kwargs 表示关键字参数，它本质上是一个 dict
def fun1(*args, **kwargs):
    print('type of args:{}'.format(type(args)))
    print('type of kwargs:{}'.format(type(kwargs)))
    print('args:{}'.format(args))
    print('kwargs:{}'.format(kwargs))


def test0():
    fun1(1, 2, 3, 4, A='a', B='b', C='c', D='d')
    '''
    type of args:<class 'tuple'>
    type of kwargs:<class 'dict'>
    args: (1, 2, 3, 4)
    kwargs: {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}
    '''


# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
def collate_fn(batch):
    # print('type of batch:', type(batch))
    # print('batch:',batch)
    '''
    type of batch: <class 'list'>
    batch: [
        (tensor([[ 0.,  2.,  4.,  6.,  8.],
        [10., 12., 14., 16., 18.],
        [20., 22., 24., 26., 28.]], dtype=torch.float64), 
        {'Y': 0, 'id': 0}), 
        
        
        (tensor([[ 1.,  3.,  5.,  7.,  9.],
        [11., 13., 15., 17., 19.],
        [21., 23., 25., 27., 29.]], dtype=torch.float64),
         {'Y': -1, 'id': 1})
    ]
    '''
    return tuple(zip(*batch))


def test_zip():
    a = [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]], [[4, 4], [4, 4]]]
    # zip函数返回zip对象
    # * 相当于解包操作
    # print('*a:', *a)
    '''
    *a: [1, 2] [3, 4] [5, 6]
    '''
    res = tuple(zip(*a))
    # print(res)
    '''
    a = [[1, 2], [3, 4], [5, 6]]
    res = ((1, 3, 5), (2, 4, 6))
    
    a = [[1, 2], [3, 4], [5, 6], [7,8]]
    res = ((1, 3, 5, 7), (2, 4, 6, 8))
    
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]]
    res = ((1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12))
    
    a = [[[1,1],[1,1]], [[2,2],[2,2]], [[3,3],[3,3]], [[4,4],[4,4]]]
    res = (([1, 1], [2, 2], [3, 3], [4, 4]), ([1, 1], [2, 2], [3, 3], [4, 4]))
    '''


class data_set(object):
    def __init__(self, datasize):
        self.X = []
        for i in range(datasize):
            self.X.append(
                np.arange(i, i + 30, 2).reshape(3, 5)
            )
        self.Y = []
        for i in range(datasize):
            self.Y.append(-i)

    def __getitem__(self, idx):
        # return self.X[idx], self.Y[idx]
        x = torch.as_tensor(self.X[idx], dtype=float)

        target = {}
        target['Y'] = self.Y[idx]
        target['id'] = idx
        return x, target

    def __len__(self):
        return len(self.X)


def getDataLoader():
    data = data_set(10)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    return data_loader


def test1():
    data_loader = getDataLoader()
    data_loader_iter = iter(data_loader)
    X, Y = next(data_loader_iter)


class data_set1(object):
    def __init__(self):
        data = np.array([
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]],
            [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]
        ], dtype=float)

        data0 = torch.tensor(data)
        data1 = torch.tensor(data)
        data2 = torch.tensor(data)
        data3 = torch.tensor(data)
        data0[:, :, 0] = 0
        data1[:, :, 0] = 1
        data2[:, :, 0] = 2
        data3[:, :, 0] = 3
        self.datas = []
        self.datas.append(data0)
        self.datas.append(data1)
        self.datas.append(data2)
        self.datas.append(data3)

    def __getitem__(self, idx):
        return torch.as_tensor(self.datas[idx])

    def __len__(self):
        return 4


# numpy和tensor的一些操作
def text_numpy_tensor():
    # 模拟三通道 5*5的图片
    data = np.array([
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]],
        [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]
    ], dtype=float)
    # print(data.shape)
    '''
    (3,5,5)
    '''
    data = torch.as_tensor(data)
    # print(data.size())
    '''
    torch.Size([3, 5, 5])
    '''
    data = np.array(data)
    data0 = torch.tensor(data)
    data1 = torch.tensor(data)
    data2 = torch.tensor(data)
    data3 = torch.tensor(data)
    data0[:, :, 0] = 0
    data1[:, :, 0] = 1
    data2[:, :, 0] = 2
    data3[:, :, 0] = 3
    datas = []
    datas.append(data0)
    datas.append(data1)
    datas.append(data2)
    datas.append(data3)  # type(datas): list
    # datas 无法用 datas = torch.as_tensor(datas)转化为Tensor对象
    # 但这个转换是必须的，因为我们将来输入模型的不是单一的数据，而是一组数据
    # 从维度上看，输入模型的数据维度描述为 (4 , 3 , 5, 5)
    # 单一的数据维度描述为 （3, 5, 5)

    # datas转换为tensor对象
    # print(type(datas[0])) # Tensor
    _datas0 = np.stack(datas) # shape (4,3,5,5)
    # print(type(_datas0)) # ndarray
    # print(type(_datas0[0])) # ndarray
    _datas0 = torch.as_tensor(_datas0)
    # print(type(_datas0)) # Tensor
    # 转换完成

    # 通过DataLoader自动批处理 将输入样本整理为一批
    dataset = data_set1()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4
    )
    data_loader_iter = iter(data_loader)
    data_from_loader = next(data_loader_iter) # data_from_loader: (4,3,3,5)
    # print(data_from_loader)
    # DataLoader获得一个batch的数据，要进行一个类似stack的操作
    # data_from_loader与stack后得到的数据相同

    # _datas1= collate_fn(datas)
    # print(_datas0.shape)
    #     # print('_datas0:',_datas0)
    #     # print('_datas1:',_datas1)



if __name__ == '__main__':
    # test0()
    # test1()
    # test_zip()
    text_numpy_tensor()
