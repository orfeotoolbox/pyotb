import os
import uuid
import multiprocessing
from collections import Counter

from pyotb.core import (App, Input,
                        get_nbchannels,
                        logger)

"""
Contains several useful functions base on pyotb
"""

def where(cond, x, y):
    """
    Functionally similar to numpy.where. Where cond is True (=1), returns x. Else returns y

    :param cond: condition, must be a raster (filepath, App, Operation...). If cond is monoband whereas x or y are
                 multiband, cond channels are expanded to match x & y ones.
    :param x: value if cond is True. Can be float, int, App, filepath, Operation...
    :param y: value if cond is False. Can be float, int, App, filepath, Operation...
    :return:
    """
    # Getting number of channels of rasters
    x_nb_channels, y_nb_channels = None, None
    if not isinstance(x, (int, float)):
        x_nb_channels = get_nbchannels(x)
    if not isinstance(y, (int, float)):
        y_nb_channels = get_nbchannels(y)

    if x_nb_channels is not None and y_nb_channels is not None:
        if x_nb_channels != y_nb_channels:
            raise Exception('X and Y images do not have the same number of bands. '
                            'X has {} bands whereas Y has {} bands'.format(x_nb_channels, y_nb_channels))

    if x_nb_channels is not None:
        x_or_y_nb_channels = x_nb_channels
    elif y_nb_channels is not None:
        x_or_y_nb_channels = y_nb_channels
    else:
        x_or_y_nb_channels = None

    # Computing the BandMathX expression of the condition
    cond_nb_channels = get_nbchannels(cond)
    # if needed, duplicate the single band binary mask to multiband to match the dimensions of x & y
    if cond_nb_channels == 1 and x_or_y_nb_channels is not None and x_or_y_nb_channels != 1:
        logger.info('The condition has one channel whereas X/Y has/have {} channels. Expanding number of channels '
                    'of condition to match the number of channels or X/Y'.format(x_or_y_nb_channels))
        cond_exp = ['im1b1'] * x_or_y_nb_channels
        cond_nb_channels = x_or_y_nb_channels
    elif cond_nb_channels != 1 and x_or_y_nb_channels is not None and cond_nb_channels != x_or_y_nb_channels:
        raise Exception('Condition and X&Y do not have the same number of bands. Condition has '
                        '{} bands whereas X&Y have {} bands'.format(cond_nb_channels, x_or_y_nb_channels))
    else:
        cond_exp = [f'im1b{b}' for b in range(1, 1 + cond_nb_channels)]

    # Computing the BandMathX expression of the  inputs
    im_count = 2
    inputs = [cond]
    if isinstance(x, (float, int)):
        x_exp = [x] * cond_nb_channels
    else:
        x_exp = [f'im{im_count}b{b}' for b in range(1, 1 + x_nb_channels)]
        im_count += 1
        inputs.append(x)
    if isinstance(y, (float, int)):
        y_exp = [y] * cond_nb_channels
    else:
        y_exp = [f'im{im_count}b{b}' for b in range(1, 1 + y_nb_channels)]
        im_count += 1
        inputs.append(y)

    # Writing the multiband expression (each band separated by a `;`)
    exp = ';'.join([f'({condition} == 1 ? {x} : {y})' for condition, x, y in zip(cond_exp, x_exp, y_exp)])
    app = App("BandMathX", il=inputs, exp=exp)

    return app


def clip(a, a_min, a_max):
    """
    Clip values of image in a range of values

    :param a: input raster, can be filepath or any pyotb object
    :param a_min: minumum value of the range
    :param a_max: maximum value of the range
    :return: raster whose values are clipped in the range
    """
    if isinstance(a, str):
        a = Input(a)

    res = where(a <= a_min, a_min,
                where(a >= a_max, a_max, a))
    return res


def define_processing_area(*args, window_rule='intersection', pixel_size_rule='minimal', interpolator='nn',
                           reference_window_input=None, reference_pixel_size_input=None):
    """
    Given several inputs, this function handles the potential resampling and cropping to same extent.
    //!\\ Not fully implemented / tested

    :param args: list of raster inputs. Can be str (filepath) or pyotb objects
    :param window_rule: Can be 'intersection', 'union', 'same_as_input', 'specify'
    :param pixel_size_rule: Can be 'minimal', 'maximal', 'same_as_input', 'specify'
    :param interpolator: Can be 'bco', 'nn', 'linear'
    :param reference_window_input: Required if window_rule = 'same_as_input'
    :param reference_pixel_size_input: Required if pixel_size_rule = 'same_as_input'
    :return: list of in-memory pyotb objects with all the same resolution, shape and extent
    """

    # Flatten all args into one list
    inputs = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            inputs.extend(arg)
        else:
            inputs.append(arg)

    # Getting metadatas of inputs
    metadatas = {}
    for input in inputs:
        if isinstance(input, str):  # this is for filepaths
            metadata = Input(input).GetImageMetaData('out')
        elif hasattr(input, 'output_parameter_key'):  # this is for Output, Input, Operation
            metadata = input.GetImageMetaData(input.output_parameter_key)
        else:  # this is for App
            metadata = input.GetImageMetaData(input.output_parameters_keys[0])
        metadatas[input] = metadata

    # Get a metadata of an arbitrary image. This is just to compare later with other images
    any_metadata = next(iter(metadatas.values()))

    # Checking if all images have the same projection
    if not all(metadata['ProjectionRef'] == any_metadata['ProjectionRef']
               for metadata in metadatas.values()):
        logger.warning('All images may not have the same CRS, which might cause unpredictable results')

    # Handling different spatial footprints
    # TODO: there seems to have a bug, ImageMetaData is not updated when running an app,
    #  cf https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2234. Should we use ImageOrigin instead?
    if not all(metadata['UpperLeftCorner'] == any_metadata['UpperLeftCorner']
               and metadata['LowerRightCorner'] == any_metadata['LowerRightCorner']
               for metadata in metadatas.values()):
        # Retrieving the bounding box that will be common for all inputs
        if window_rule == 'intersection':
            # The coordinates depend on the orientation of the axis of projection
            if any_metadata['GeoTransform'][1] >= 0:
                ULX = max(metadata['UpperLeftCorner'][0] for metadata in metadatas.values())
                LRX = min(metadata['LowerRightCorner'][0] for metadata in metadatas.values())
            else:
                ULX = min(metadata['UpperLeftCorner'][0] for metadata in metadatas.values())
                LRX = max(metadata['LowerRightCorner'][0] for metadata in metadatas.values())
            if any_metadata['GeoTransform'][-1] >= 0:
                LRY = min(metadata['LowerRightCorner'][1] for metadata in metadatas.values())
                ULY = max(metadata['UpperLeftCorner'][1] for metadata in metadatas.values())
            else:
                LRY = max(metadata['LowerRightCorner'][1] for metadata in metadatas.values())
                ULY = min(metadata['UpperLeftCorner'][1] for metadata in metadatas.values())

        elif window_rule == 'same_as_input':
            ULX = metadatas[reference_window_input]['UpperLeftCorner'][0]
            LRX = metadatas[reference_window_input]['LowerRightCorner'][0]
            LRY = metadatas[reference_window_input]['LowerRightCorner'][1]
            ULY = metadatas[reference_window_input]['UpperLeftCorner'][1]
        elif window_rule == 'specify':
            pass
            # TODO : it is when the user explicitely specifies the bounding box -> add some arguments in the function
        elif window_rule == 'union':
            pass
            # TODO : it is when the user wants the final bounding box to be the union of all bounding box
            #  It should replace any 'outside' pixel by some NoData -> add `fillvalue` argument in the function

        logger.info(
            'Cropping all images to extent Upper Left ({}, {}), Lower Right ({}, {}) '.format(ULX, ULY, LRX, LRY))

        # Applying this bounding box to all inputs
        new_inputs = []
        for input in inputs:
            try:
                new_input = App('ExtractROI', {'in': input, 'mode': 'extent', 'mode.extent.unit': 'phy',
                                               'mode.extent.ulx': ULX, 'mode.extent.uly': LRY,  # bug in OTB <= 7.3 :
                                               'mode.extent.lrx': LRX, 'mode.extent.lry': ULY})  # ULY/LRY are inverted
                # TODO: OTB 7.4 fixes this bug, how to handle different versions of OTB?
                new_inputs.append(new_input)
                # Potentially update the reference inputs for later resampling
                if str(input) == str(reference_pixel_size_input):  # we use comparison of string because calling '=='
                    # on pyotb objects underlyingly calls BandMathX application, which is not desirable
                    reference_pixel_size_input = new_input
            except Exception as e:
                logger.error(e)
                logger.error('Images may not intersect : {}'.format(input))
                # TODO: what should we do then? return an empty raster ? fail ? return None ?
        inputs = new_inputs

        # Update metadatas
        metadatas = {input: input.GetImageMetaData('out') for input in inputs}

    # Get a metadata of an arbitrary image. This is just to compare later with other images
    any_metadata = next(iter(metadatas.values()))

    # Handling different pixel sizes
    if not all(metadata['GeoTransform'][1] == any_metadata['GeoTransform'][1]
               and metadata['GeoTransform'][5] == any_metadata['GeoTransform'][5]
               for metadata in metadatas.values()):
        # Retrieving the pixel size that will be common for all inputs
        if pixel_size_rule == 'minimal':
            # selecting the input with the smallest x pixel size
            reference_input = min(metadatas, key=lambda x: metadatas[x]['GeoTransform'][1])
        if pixel_size_rule == 'maximal':
            # selecting the input with the highest x pixel size
            reference_input = max(metadatas, key=lambda x: metadatas[x]['GeoTransform'][1])
        elif pixel_size_rule == 'same_as_input':
            reference_input = reference_pixel_size_input
        elif pixel_size_rule == 'specify':
            pass
            # TODO : when the user explicitely specify the pixel size -> add argument inside the function

        logger.info('Resampling all inputs to resolution : {}'.format(metadatas[reference_input]['GeoTransform'][1]))

        # Perform resampling on inputs that do not comply with the target pixel size
        new_inputs = []
        for input in inputs:
            if metadatas[input]['GeoTransform'][1] != metadatas[reference_input]['GeoTransform'][1]:
                superimposed = App('Superimpose', inr=reference_input, inm=input, interpolator=interpolator)
                new_inputs.append(superimposed)
            else:
                new_inputs.append(input)
        inputs = new_inputs

        # Update metadatas
        metadatas = {input: input.GetImageMetaData('out') for input in inputs}

    # Final superimposition to be sure to have the exact same image sizes
    # Getting the sizes of images
    image_sizes = {}
    for input in inputs:
        if isinstance(input, str):
            input = Input(input)
        image_sizes[input] = input.shape[:2]

    # Selecting the most frequent image size. It will be used as reference.
    most_common_image_size, _ = Counter(image_sizes.values()).most_common(1)[0]
    same_size_images = [input for input, image_size in image_sizes.items() if image_size == most_common_image_size]

    # Superimposition for images that do not have the same size as the others
    new_inputs = []
    for input in inputs:
        if image_sizes[input] != most_common_image_size:
            new_input = App('Superimpose', inr=same_size_images[0], inm=input, interpolator=interpolator)
            new_inputs.append(new_input)
        else:
            new_inputs.append(input)
    inputs = new_inputs

    return inputs


def run_tf_function(func):
    """
    This decorator enables using a function that calls some TF operations, with pyotb object as inputs.

    For example, you can write a function that uses TF operations like this :
        @run_tf_function
        def multiply(input1, input2):
            import tensorflow as tf
            return tf.multiply(input1, input2)

    Then you can use it like this :
        result = multiply(pyotb_object1, pyotb_object1)  # this is a pyotb object

    :param func: function taking one or several inputs and returning *one* output
    :return wrapper: a function that returns a pyotb object
    """

    def create_and_save_tf_model(output_dir, *inputs):
        """
        Simply creates the TF model and save it to temporary location.
        //!\\ Currently incompatible with OTBTF, to be run inside a multiprocessing.Process
        //!\\ Does not work if OTBTF has been used previously in the script

        :param output_dir: directory under which to save the model
        :param inputs: a list of pyotb objects or int/float
        """
        import tensorflow as tf
        # Change the raster inputs to TF inputs
        model_inputs = []  # model inputs corresponding to rasters
        tf_inputs = []  # inputs for the TF function corresponding to all inputs (rasters, int...)
        for input in inputs:
            if not isinstance(input, (int, float)):
                nb_bands = input.shape[-1]
                input = tf.keras.Input((None, None, nb_bands))
                model_inputs.append(input)
            tf_inputs.append(input)

        # call the TF operations on theses inputs
        output = func(*tf_inputs)

        # Create and save the .pb model
        model = tf.keras.Model(inputs=model_inputs, outputs=output)
        model.save(output_dir)

    def wrapper(*inputs, tmp_dir='/tmp'):
        """
        For the user point of view, this function simply applies some TF operations to some rasters.
        Underlyingly, it saveq a .pb model that describe the TF operations, then creates an OTB ModelServe application
        that applies this .pb model to the actual inputs.

        :param inputs: a list of pyotb objects, filepaths or int/float numbers
        :param tmp_dir: directory where temporary models can be written
        :return: a pyotb object, output of TensorFlowModelServe
        """
        # Change potential string filepaths to pyotb objects
        inputs = [Input(input) if isinstance(input, str) and not isinstance(input, (int, float)) else input for input in
                  inputs]

        # Create and save the model. This is executed **inside an independent process** because (as of 2021-11),
        # tensorflow python library and OTBTF are incompatible
        out_savedmodel = os.path.join(tmp_dir, 'tmp_otbtf_model_{}'.format(uuid.uuid4()))
        p = multiprocessing.Process(target=create_and_save_tf_model, args=(out_savedmodel, *inputs,))
        p.start()
        p.join()

        # Getting the nb of inputs and setting it for OTBTF
        raster_inputs = [input for input in inputs if not isinstance(input, (int, float))]
        nb_model_inputs = len(raster_inputs)
        os.environ['OTB_TF_NSOURCES'] = str(nb_model_inputs)

        # Run the OTBTF model serving application
        model_serve = App('TensorflowModelServe',
                          {'model.dir': out_savedmodel,
                           'optim.disabletiling': 'on', 'model.fullyconv': 'on'}, execute=False)

        for i, input in enumerate(raster_inputs):
            model_serve.set_parameters({'source{}.il'.format(i + 1): [input]})

        model_serve.Execute()
        # TODO: handle the deletion of the temporary model ?

        return model_serve

    return wrapper
