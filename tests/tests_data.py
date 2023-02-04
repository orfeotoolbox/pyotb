import pyotb

FILEPATH = "/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif"
INPUT = pyotb.Input(FILEPATH)

TEST_IMAGE_STATS = {
    'out.mean': [79.5505, 109.225, 115.456, 249.349],
    'out.min': [33, 64, 91, 47],
    'out.max': [255, 255, 230, 255],
    'out.std': [51.0754, 35.3152, 23.4514, 20.3827]
}

SIMPLE_SERIALIZATION = {'name': 'ManageNoData',
                        'parameters': {'in': {'name': 'OrthoRectification',
                            'parameters': {'io.in': {'name': 'BandMath',
                                'parameters': {'il': ['/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif'],
                                'exp': 'im1b1', 'ram': 256}},
                            'map.utm.zone': 31, 'map.utm.northhem': False, 'map.epsg.code': 4326, 'outputs.isotropic': True, 'outputs.default': 0.0,
                            'elev.default': 0.0, 'interpolator.bco.radius': 2, 'opt.rpc': 10, 'opt.ram': 256,
                            'opt.gridspacing': 4.0, 'outputs.ulx': 560000.8125, 'outputs.uly': 5495732.5, 'outputs.sizex': 251, 'outputs.sizey': 304,
                            'outputs.spacingx': 5.997312068939209, 'outputs.spacingy': -5.997312068939209, 'outputs.lrx': 561506.125, 'outputs.lry': 5493909.5}},
                        'usenan': False, 'mode.buildmask.inv': 1.0, 'mode.buildmask.outv': 0.0, 'mode.changevalue.newv': 0.0, 'mode.apply.ndval': 0.0, 'ram': 256}}

COMPLEX_SERIALIZATION = {
    'name': 'BandMathX',
    'parameters': {'il': [{'name': 'OrthoRectification',
        'parameters': {'io.in': {'name': 'BandMath',
            'parameters': {
                'il': ['/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif'],
                'exp': 'im1b1', 'ram': 256}},
        'map.utm.zone': 31, 'map.utm.northhem': False, 'map.epsg.code': 4326, 'outputs.isotropic': True, 'outputs.default': 0.0, 'elev.default': 0.0,
        'interpolator.bco.radius': 2, 'opt.rpc': 10, 'opt.ram': 256, 'opt.gridspacing': 4.0, 'outputs.ulx': 560000.8125, 'outputs.uly': 5495732.5,
        'outputs.sizex': 251, 'outputs.sizey': 304, 'outputs.spacingx': 5.997312068939209, 'outputs.spacingy': -5.997312068939209, 'outputs.lrx': 561506.125, 'outputs.lry': 5493909.5}},
    {'name': 'ManageNoData',
        'parameters': {'in': {'name': 'OrthoRectification',
        'parameters': {'io.in': {'name': 'BandMath',
            'parameters': {'il': ['/vsicurl/https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/raw/develop/Data/Input/SP67_FR_subset_1.tif'],
            'exp': 'im1b1', 'ram': 256}},
        'map.utm.zone': 31, 'map.utm.northhem': False, 'map.epsg.code': 4326, 'outputs.isotropic': True, 'outputs.default': 0.0, 'elev.default': 0.0, 'interpolator.bco.radius': 2,
        'opt.rpc': 10, 'opt.ram': 256, 'opt.gridspacing': 4.0, 'outputs.ulx': 560000.8125, 'outputs.uly': 5495732.5,
        'outputs.sizex': 251, 'outputs.sizey': 304, 'outputs.spacingx': 5.997312068939209,
        'outputs.spacingy': -5.997312068939209, 'outputs.lrx': 561506.125, 'outputs.lry': 5493909.5}},
    'usenan': False, 'mode.buildmask.inv': 1.0, 'mode.buildmask.outv': 0.0, 'mode.changevalue.newv': 0.0, 'mode.apply.ndval': 0.0,  'ram': 256}}],
    'exp': 'im1+im2', 'ram': 256}
}
