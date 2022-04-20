import pandas as pd
import matplotlib.pyplot as plt


with open('log.csv', 'r') as f:
    log_data = pd.read_csv(f, header=None)

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(log_data[0], log_data[1], label='Training loss')
ax.plot(log_data[0], log_data[2], label='Testing loss')
plt.savefig('loss_2.png')
plt.show()
