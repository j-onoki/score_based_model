import NCSN 
import function as f
import torch
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NCSN.UNet().to(device)
    
    model.load_state_dict(torch.load("model1000.pth"))

    #MNISTデータのダウンロード
    train_images = torch.load("./data/train.pt")
    test_images = torch.load("./data/test.pt")
    train_labels = torch.load("./data/train_label.pt")
    test_labels = torch.load("./data/test_label.pt")

    # #データセットの作成
    # train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    # test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    # x = test_dataset[0][0].reshape(1,1,32,32).to(device)
    noise_image = test_images[0] + math.sqrt(1)*torch.randn(size=test_images[0].size())
    
    x0 = noise_image
    with torch.no_grad(): 
        xt = f.sampling(100, 0.0005, 32, 32, model, device, x0)
    
    plt.figure()
    plt.imshow(xt.cpu().reshape(32, 32), cmap = "gray")
    plt.savefig('./image/test_xt_T100_e00005.png')
    
    print(xt)
    
    
    # plt.figure()
    # plt.imshow(test_images[0].cpu().reshape(32, 32), cmap = "gray")
    # plt.savefig('./image/test.png')
    
    # plt.figure()
    # plt.imshow(noise_image.cpu().reshape(32, 32), cmap = "gray")
    # plt.savefig('./image/test_noise.png')
    
    print(torch.mean(noise_image))
    