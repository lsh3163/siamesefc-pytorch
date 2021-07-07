import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
import os
video_list = os.listdir("./result")
for video in video_list:
    l = len(os.listdir("./result/{}".format(video)))
    print(l, video)
    path = [f"./result/{{}}/{i}.jpg".format(video) for i in range(1, l)]
    paths = [ Image.open(i) for i in path]

    imageio.mimsave('./gif/{}.gif'.format(video), paths, fps=30)