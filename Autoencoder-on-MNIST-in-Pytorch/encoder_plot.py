from Autoencoder import Autoencoder, testset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

results = pd.DataFrame()
nums = []
model = Autoencoder().cpu()
model.load_state_dict(torch.load('Models/model_classic.pt'))
model.eval()

i = 0
for data in testset:
    if i < 300:
        img, num = data
        img = img.reshape([1, 1, 28, 28])
        output = model.encoder(img).detach().numpy()
        nums.append(num)
        results = results.append([output[0, :]])
        i += 1
    else:
        break

results = results.reset_index(drop=True)
results = pd.concat([results, pd.Series(nums)], axis=1, ignore_index=True)
results.to_csv('encoder.csv', header=False, index=False)

# fig, ax = plt.subplots()
# im = plt.scatter(results[0], results[1], c=results[2], cmap=cm.tab10)
# plt.colorbar()
# plt.show()
