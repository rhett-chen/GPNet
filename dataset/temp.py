import OpenEXR
import Imath
import numpy as np
import cv2
import sys
sys.path.append('/home/zibo/GPNet/tools/')
sys.path.append("/home/zibo/GPNet")

import tools.depth2points as dp
File = OpenEXR.InputFile('/home/zibo/GPNet/GPNet_release_data/images/cylinder010/render0Depth0001.exr')
PixType = Imath.PixelType(Imath.PixelType.FLOAT)
DW = File.header()['dataWindow']
Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
r = np.reshape(rgb[0], (Size[1], Size[0]))
print(r)
cv2.imwrite('/home/zibo/GPNet/dataset/te.jpg', dp.encode_depth_to_image(r))
