import pyotb

img_pth = "/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/" \
          "otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif?inline=false"
bm = pyotb.BandMath({"il": [img_pth], "exp": "im1b1"})
bm.write("/tmp/toto.tif")  # Comment this line --> Works
print(bm.shape)