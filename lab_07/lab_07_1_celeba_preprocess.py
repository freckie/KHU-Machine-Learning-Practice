# 178, 218 to 28, 28

import os
from PIL import Image
from tqdm import tqdm

path_dir = './celeba_data/raw/'
target_dir = './celeba_data/processed/'
files = os.listdir(path_dir)
files.sort()

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for idx, it in tqdm(enumerate(files), ncols=90, desc='Processing'):
    img = Image.open(path_dir + it)
    img.resize((28, 28)).save(target_dir + it)
