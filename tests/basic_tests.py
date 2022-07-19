import pyotb

filepath = 'image.tif'
inp = pyotb.Input(filepath)

assert inp.shape == (304, 251, 4)

# Test slicer
extract = inp[:50, :60, :3]
assert extract.shape == (50, 60, 3)

# Test ReadImageInfo
info = pyotb.ReadImageInfo(inp, quiet=True)
assert info.sizex == 251
assert info.sizey == 304
assert info['numberbands'] == info.numberbands == 4

# Test Statistics
stats = pyotb.ComputeImagesStatistics(il=inp, quiet=True)
assert stats['out.min'] == "[33, 64, 91, 47]"

# Test Statistics on a Slicer
slicer_stats = pyotb.ComputeImagesStatistics(il=[inp[:10, :10, 0]], quiet=True)
assert slicer_stats['out.min'] == '[180]'

