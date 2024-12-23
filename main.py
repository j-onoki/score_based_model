import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import NCSN
import load_mnist as l
import learning
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #GPUが使えるか確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    #モデルのインスタンス化
    model = NCSN.UNet().to(device)
    print(model)
    #model.load_state_dict(torch.load("model1.pth"))

    #MNISTデータのダウンロード
    train_images = torch.load("./data/train.pt")
    test_images = torch.load("./data/test.pt")
    train_labels = torch.load("./data/train_label.pt")
    test_labels = torch.load("./data/test_label.pt")
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    #ミニバッチの作成
    train_loader, test_loader = l.loader_MNIST(train_dataset, test_dataset)

    #最適化法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=0.00001, amsgrad=True)

    num_epochs = 1000
    train_loss_list, test_loss_list = learning.lerning(model, train_loader, test_loader, optimizer, num_epochs, device)

    plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('./image/loss.png')

    #モデルを保存する。
    torch.save(model.state_dict(), "model.pth")