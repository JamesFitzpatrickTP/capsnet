import numpy as np
import matplotlib.pyplot as plt
import torch

from a3dsurf.simple_process import load_volume
from a3dshow import display

import layers

base = '/home/jfitzpatrick/a3d/dicomstacks/ExtraData/P00019'
img_tail = '/manual_print_data/segmentation/ml/00_original/'
lab_tail = '/manual_print_data/segmentation/ml/05_hollow/'

imgs = load_volume({
    "ref_type": "path",
    "ref": base + img_tail})

labs = load_volume({
    "ref_type": "path",
    "ref": base + lab_tail})


img_max = imgs.vol.max()
img_min = imgs.vol.min()
img_dif = img_max - img_min

imgs = (imgs.vol - img_min) / img_dif
labs = (labs.vol > 0.1).astype(int)

model = layers.DownNetwork()
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()


def to_tensor(vol, idx, expand=True):
    ten = vol[..., idx]
    if expand:
        ten = np.expand_dims(ten, 0)
        ten = np.expand_dims(ten, 0)
    return torch.tensor(ten).float()
    

for epoch in range(2):
    print('###### EPOCH {} ######'.format(epoch))
    for idx in range(40, 61):
        img = to_tensor(imgs, idx, expand=True)
        lab = to_tensor(labs, idx, expand=False)
        pred = model(img)
        loss = criterion(pred, lab)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        print("##### ERROR: ", loss, ' ######')
        plt.imshow(pred.data.numpy())
        plt.show()

        
