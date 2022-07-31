from Autoencoder import Autoencoder, testset
import matplotlib.pyplot as plt
import torch

model = Autoencoder().cpu()
model.load_state_dict(torch.load('Models/model_e19.pt'))
model.eval()

n = [247]

for data in testset:
    img, num = data
    plt.imshow(img.reshape([28, 28]))
    print(num)
    plt.show()

    img = img.reshape([1, 1, 28, 28])
    img = torch.autograd.Variable(img).cpu()
    output = model(img)
    plt.imshow(output.detach().numpy().reshape([28, 28]))
    plt.show()
