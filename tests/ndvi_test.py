import pyotb

filepath = 'Data/Input/QB_MUL_ROI_1000_100.tif'
inp = pyotb.Input(filepath)

# Compute NDVI with bandmath
ndvi_bandmath = (inp[:, :, -1] - inp[:, :, [2]]) / (inp[:, :, -1] + inp[:, :, [2]])
assert ndvi_bandmath.exp == '((im1b4 - im1b3) / (im1b4 + im1b3))'
ndvi_bandmath.write('/tmp/ndvi_bandmath.tif', pixel_type='float')

# Compute NDVI with RadiometricIndices app
ndvi_indices = pyotb.RadiometricIndices({'in': inp, 'list': 'Vegetation:NDVI',
                                         'channels.red': 3, 'channels.nir': 4})
ndvi_indices.write('/tmp/ndvi_indices.tif', pixel_type='float')

compared = pyotb.CompareImages({'ref.in': ndvi_indices, 'meas.in': '/tmp/ndvi_bandmath.tif'})
assert compared.count == 0
assert compared.mse == 0

# Threshold
thresholded_indices = pyotb.where(ndvi_indices >= 0.3, 1, 0)
thresholded_bandmath = pyotb.where(ndvi_bandmath >= 0.3, 1, 0)
assert thresholded_indices.exp == '((im1b1 >= 0.3) ? 1 : 0)'
assert thresholded_bandmath.exp == '((((im1b4 - im1b3) / (im1b4 + im1b3)) >= 0.3) ? 1 : 0)'

# Sum of bands
summed = sum(inp[:, :, b] for b in range(inp.shape[-1]))
assert summed.exp == '((((0 + im1b1) + im1b2) + im1b3) + im1b4)'
