import pyotb

img_pth = "/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/" \
          "otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif?inline=false"
bm = pyotb.MeanShiftSmoothing({"in": img_pth, "fout": "/tmp/toto.tif", "foutpos": "/tmp/titi.tif"})
print(bm.find_outputs())