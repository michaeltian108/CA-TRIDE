import torchvision


'''
Download MNIST and Fashion-MNIST datasets
'''

def md_downloader():
    print('>> Checking MNIST ...')
    torchvision.datasets.MNIST('~/rob_IR/datasets/', download=True)

    print('>> Checking FashionMNIST ...')
    torchvision.datasets.FashionMNIST('~/rob_IR/datasets/', download=True)

    print('>> Done!')