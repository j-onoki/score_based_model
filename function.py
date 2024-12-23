import torch

L = 10


def addNoise(images, device):
    sigma = torch.logspace(start=0, end=-2, steps=L, base=10).to(device)
    l = torch.randint(L, size=(images.size()[0],)).to(device)
    sigmal = sigma[l]
    output = torch.zeros(size=images.size()).to(device)
    dim1 = images.size()[2]
    dim2 = images.size()[3]

    for i in range(images.size()[0]):
        output[i, 0, :] = images[i, 0, :] + sigmal[i] * torch.randn(size=(dim1, dim2)).to(device)

    return output, sigmal


def criterion(images, inputs, outputs, sigma):
    sigma = sigma.reshape(sigma.size()[0], 1, 1, 1)
    temp = torch.mul(outputs, sigma)
    temp2 = (images - inputs) / sigma
    loss = torch.nn.functional.mse_loss(temp, temp2)
    return loss


def sampling(T, epsilon, dim1, dim2, model, device, x0):
    sigma = torch.logspace(start=0, end=-2, steps=L, base=10).to(device)
    # x0 = sigma[0]*torch.randn(size=(1, 1, dim1, dim2)).to(device)

    xt = x0.reshape(1, 1, dim1, dim2).to(device)
    for i in range(L):
        alphai = epsilon * ((sigma[i] ** 2) / (sigma[L - 1] ** 2))
        for t in range(T):
            zt = torch.randn(size=(dim1, dim2)).to(device)
            score = model(xt, sigma[i] * torch.ones((1, xt.size[0])))
            if torch.any(torch.isnan(score)):
                print("score has nan")
            xt = xt + alphai * score / 2 + torch.sqrt(alphai) * zt
            if torch.any(torch.isnan(xt)):
                print(f"nandetayo L={i}, T={t} alphai={alphai}")
        print(i)
    return xt


if __name__ == "__main__":
    s = torch.ones(size=(10, 1, 5, 5))
    sigma = torch.randint(1, 10, size=(10, 1, 1, 1))
    a = s / sigma
    anorm = torch.norm(a) ** 2 / 10
    print(s)
    print(sigma)
    print(a)
    print(anorm)
