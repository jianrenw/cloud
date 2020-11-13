import os
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot
import pickle
from matplotlib.pyplot import imshow

with load('intermediate_states.npz') as data:
    state_emb_list = data['state_embedding']
    X = np.array(state_emb_list)
X_embedded = TSNE(n_components=2).fit_transform(X)

filePath = "pathToYourDataDir"
# stage img
images = os.listdir(filePath)

tx, ty = X_embedded[:,0], X_embedded[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 3000
height = 3000
max_dim = 300

full_image = Image.new('RGBA', (width, height))
cnt = 0
for img, x, y in zip(images, tx, ty):
    cnt += 1
    if cnt % 3 == 0:
        continue
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs) * 3, int(tile.height/rs) * 3), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

matplotlib.pyplot.figure(figsize = (64,64))
imshow(full_image)

full_image.save("examplex-tSNE-stateembedding.png")

# flatten

import rasterfairy

nx = 40
ny = 20

# assign to grid
grid_assignment = rasterfairy.transformPointCloud2D(X_embedded[:800], target=(nx, ny))
tile_width = 64
tile_height = 64

full_width = tile_width * nx
full_height = tile_height * ny
aspect_ratio = float(tile_width) / tile_height

grid_image = Image.new('RGB', (full_width, full_height))

for img, grid_pos in zip(images[:800], grid_assignment[0]):
    idx_x, idx_y = grid_pos
    x, y = tile_width * idx_x, tile_height * idx_y
    tile = Image.open(img)
    tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
    if (tile_ar > aspect_ratio):
        margin = 0.5 * (tile.width - aspect_ratio * tile.height)
        tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
    else:
        margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
        tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
    tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
    grid_image.paste(tile, (int(x), int(y)))

matplotlib.pyplot.figure(figsize = (16,12))
imshow(grid_image)

grid_image.save("examplex-tSNE-stateembedding-grid.png")