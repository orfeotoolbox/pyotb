import multiprocessing

import otbApplication
from collections import Counter
import logging
import sys
import os
from abc import ABC
import numpy as np
import uuid

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


class otbObject(ABC):
    """Gathers common operations for any OTB in-memory raster"""
    def __getitem__(self, key):
        """
        This function enables 2 things :
        - access attributes like that : object['any_attribute']
        - slicing, i.e. selecting ROI/bands. For example, selecting first 3 bands: object[:, :, :3]
                                                          selecting bands 1, 2 & 5 : object[:, :, [0, 1, 4]]
                                                          selecting 1000x1000 subset : object[:1000, :1000]
        """
        # Accessing string attributes
        if isinstance(key, str):
            return self.__dict__.get(key)

        # Slicing
        elif not isinstance(key, tuple) or (isinstance(key, tuple) and len(key) < 2):
            raise ValueError('`{}`cannot be interpreted as valid slicing. Slicing should be 2D or 3D.'.format(key))
        elif isinstance(key, tuple) and len(key) == 2:
            # adding a 3rd dimension
            key = key + (slice(None, None, None),)
        (rows, cols, channels) = key

        # Initialize the app that will be used for slicing
        app = App('ExtractROI', {"in": self, 'mode': 'extent'})

        # Channel slicing
        nb_channels = get_nbchannels(self)
        if channels != slice(None, None, None):
            # if needed, converting int to list
            if isinstance(channels, int):
                channels = [channels]
            # if needed, converting slice to list
            elif isinstance(channels, slice):
                channels_start = channels.start if channels.start is not None else 0
                channels_end = channels.stop if channels.stop is not None else nb_channels
                channels_step = channels.step if channels.step is not None else 1
                channels = range(channels_start, channels_end, channels_step)
            elif not isinstance(channels, list):
                raise ValueError(
                    'Invalid type for channels, should be int, slice or list of bands. : {}'.format(channels))

            # Change the potential negative index values to reverse index
            channels = [c if c >= 0 else nb_channels + c for c in channels]

            app.set_parameters(cl=[f'Channel{i+1}' for i in channels])

        # Spatial slicing
        # TODO: handle PixelValue app so that accessing value is possible, e.g. obj[120, 200, 0]
        # TODO TBD: handle the step value in the slice so that nn undersampling is possible ? e.g. obj[::2, ::2]
        if rows.start is not None:
            app.set_parameters({'mode.extent.uly': rows.start})
        if rows.stop is not None and rows.stop != -1:
            app.set_parameters({'mode.extent.lry': rows.stop - 1})  # subtract 1 to be compliant with python convention
        if cols.start is not None:
            app.set_parameters({'mode.extent.ulx': cols.start})
        if cols.stop is not None and cols.stop != -1:
            app.set_parameters({'mode.extent.lrx': cols.stop - 1})  # subtract 1 to be compliant with python convention

        return app

    @property
    def shape(self):
        """
        Enables to retrieve the shape of a pyotb object. Can not be called before app.Execute()
        :return shape: (width, height, bands)
        """

        if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation
            output_parameter_key = self.output_parameter_key
        else:  # this is for App
            output_parameter_key = self.get_output_parameters_keys()[0]
        image_size = self.GetImageSize(output_parameter_key)
        image_bands = self.GetImageNbBands(output_parameter_key)
        # TODO: it currently returns (width, height, bands), should we use numpy convention (height, width, bands) ?
        return *image_size, image_bands

    def __getattr__(self, name):
        """This method is called when the default attribute access fails. We choose to try to access the attribute of
        self.app. Thus, any method of otbApplication can be used transparently in the wrapper,
        e.g. SetParameterOutputImagePixelType() or ExportImage() work"""
        return getattr(self.app, name)

    def write(self, *args, filename_extension=None, pixel_type=None, is_intermediate=False, **kwargs):
        """
        Write the output
        :param args: Can be : - dictionary containing key-arguments enumeration. Useful when a key contains
                                non-standard characters such as a point, e.g. {'io.out':'output.tif'}
                              - string, useful when there is only one output, e.g. 'output.tif'
        :param filename_extension: Optional, an extended filename as understood by OTB (e.g. "&gdal:co:TILED=YES")
                                   Will be used for all outputs
        :param pixel_type: Can be : - dictionary {output_parameter_key: pixeltype} when specifying for several outputs
                                    - str (e.g. 'uint16') or otbApplication.ImagePixelType_... When there are several
                                      outputs, all outputs are writen with this unique type
                           Valid pixel types are double, float, uint8, uint16, uint32, int16, int32, float, double,
                           cint16, cint32, cfloat, cdouble.
        :param is_intermediate: WARNING: not fully tested. whether the raster we want to write is intermediate,
                                i.e. not the final result of the whole pipeline
        :param kwargs: keyword arguments e.g. out='output.tif'
        """

        # Gather all input arguments in kwargs dict
        if args:
            for arg in args:
                if isinstance(arg, dict):
                    kwargs.update(arg)
                elif isinstance(arg, str) and kwargs:
                    logging.warning('Keyword arguments specified, ignoring argument: {}'.format(arg))
                elif isinstance(arg, str):
                    if hasattr(self, 'output_parameter_key'):  # this is for Output, Operation
                        output_parameter_key = self.output_parameter_key
                    else:  # this is for App
                        output_parameter_key = self.get_output_parameters_keys()[0]
                    kwargs.update({output_parameter_key: arg})

        # Handling pixel types
        pixel_types = {}
        if isinstance(pixel_type, str):  # this correspond to 'uint8' etc...
            pixel_type = getattr(otbApplication, 'ImagePixelType_{}'.format(pixel_type))
            pixel_types = {param_key: pixel_type for param_key in kwargs}
        elif isinstance(pixel_type, int):  # this corresponds to ImagePixelType_...
            pixel_types = {param_key: pixel_type for param_key in kwargs}
        elif isinstance(pixel_type, dict):  # this is to specify a different pixel type for each output
            for key, this_pixel_type in pixel_type.items():
                if isinstance(this_pixel_type, str):
                    this_pixel_type = getattr(otbApplication, 'ImagePixelType_{}'.format(this_pixel_type))
                if isinstance(this_pixel_type, int):
                    pixel_types[key] = this_pixel_type

        if kwargs:
            # Handling the writing of intermediary outputs. Not extensively tested.
            if is_intermediate:
                self.app.PropagateConnectMode(False)

            for output_parameter_key, output_filename in kwargs.items():
                out_fn = output_filename
                if filename_extension:
                    if not out_fn.endswith('?'):
                        out_fn += "?"
                    out_fn += filename_extension
                    logging.info("Using extended filename for output.")
                logging.info("Write output for \"{}\" in {}".format(output_parameter_key, out_fn))
                self.app.SetParameterString(output_parameter_key, out_fn)

                if output_parameter_key in pixel_types:
                    self.app.SetParameterOutputImagePixelType(output_parameter_key, pixel_types[output_parameter_key])

            self.app.ExecuteAndWriteOutput()

    def __add__(self, other):
        """Overrides the default addition and flavours it with BandMathX"""
        return Operation('+', self, other)

    def __sub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('-', self, other)

    def __mul__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('*', self, other)

    def __truediv__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('/', self, other)

    def __radd__(self, other):
        """Overrides the default reverse addition and flavours it with BandMathX"""
        return Operation('+', other, self)

    def __rsub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('-', other, self)

    def __rmul__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('*', other, self)

    def __rtruediv__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        return Operation('/', other, self)

    def __ge__(self, other):
        """Overrides the default greater or equal and flavours it with BandMathX"""
        return Operation('>=', self, other)

    def __le__(self, other):
        """Overrides the default greater or equal and flavours it with BandMathX"""
        return Operation('<=', self, other)

    def __gt__(self, other):
        """Overrides the default greater operator and flavours it with BandMathX"""
        return Operation('>', self, other)

    def __lt__(self, other):
        """Overrides the default less operator and flavours it with BandMathX"""
        return Operation('<', self, other)

    def __eq__(self, other):
        """Overrides the default eq operator and flavours it with BandMathX"""
        return Operation('==', self, other)

    def __ne__(self, other):
        """Overrides the default different operator and flavours it with BandMathX"""
        return Operation('!=', self, other)

    def __or__(self, other):
        return Operation('|', self, other)

    def __and__(self, other):
        return Operation('&', self, other)

    def __abs__(self):
        return Operation('abs', self)

    # TODO: other operations ?
    #  e.g. __pow__... cf https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __hash__(self):
        return id(self)

    def __array__(self):
        """
        This is called when running np.asarray(pyotb_object)
        :return: a numpy array
        """
        if hasattr(self, 'output_parameter_key'):  # this is for Output, Operation
            output_parameter_key = self.output_parameter_key
        else:  # this is for App
            output_parameter_key = self.output_parameters_keys[0]
        return self.app.ExportImage(output_parameter_key)['array']

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        This is called whenever a numpy function is called on a pyotb object.

        :param ufunc: numpy function
        :param method: a internal numpy argument
        :param inputs: input, at least one being pyotb object. If there are several pyotb objects, they must all have
                       the same georeference and pixel size.
        :param kwargs: some numpy kwargs
        :return:
        """
        if method == '__call__':
            # Converting potential pyotb inputs to arrays
            arrays = []
            image_dic = None
            for input in inputs:
                if isinstance(input, (float, int, np.ndarray, np.generic)):
                    arrays.append(input)
                elif isinstance(input, (App, Input, Output, Operation)):
                    if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation
                        output_parameter_key = self.output_parameter_key
                    else:  # this is for App
                        output_parameter_key = self.output_parameters_keys[0]
                    image_dic = input.app.ExportImage(output_parameter_key)
                    array = image_dic['array']
                    arrays.append(array)
                else:
                    print(type(self))
                    return NotImplemented

            # Performing the numpy operation
            result_array = ufunc(*arrays, **kwargs)
            result_dic = image_dic
            result_dic['array'] = result_array

            # Importing back to OTB
            app = App('ExtractROI', image_dic=result_dic, execute=False)
            if result_array.shape[2] == 1:
                app.ImportImage('in', result_dic)
            else:
                app.ImportVectorImage('in', result_dic)
            app.Execute()
            return app

        else:
            return NotImplemented


class Input(otbObject):
    """
    Class for transforming a filepath to pyOTB object
    """
    def __init__(self, filepath):
        self.app = App('ExtractROI', filepath).app
        self.output_parameter_key = 'out'
        self.filepath = filepath

    def __str__(self):
        return '<pyotb.Input object from {}>'.format(self.filepath)


class Output(otbObject):
    """
    Class for output of an app
    """
    def __init__(self, app, output_parameter_key):
        self.app = app  # keeping a reference of the app
        self.output_parameter_key = output_parameter_key

    def __str__(self):
        return '<pyotb.Output {} object, id {}>'.format(self.app.GetName(), id(self))


class App(otbObject):
    """
    Class of an OTB app
    """
    def __init__(self, appname, *args, execute=True, image_dic=None, **kwargs):
        """
        Enables to run an otb app as a oneliner. Handles in-memory connection between apps
        :param appname: name of the app, e.g. 'Smoothing'
        :param args: Can be : - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                                (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                              - string, App or Output, useful when the user wants to specify the input "in"
                              - list, useful when the user wants to specify the input list 'il'
        :param execute: whether to Execute the app. Should be True when creating an app as a oneliner. False is
                                                    advisable when not all mandatory params are set at initialization
        :param image_dic: optional. Enables to keep a reference to image_dic. image_dic is a dictionary, such as
                          the result of app.ExportImage(). Use it when the app takes a numpy array as input.
                          See this related issue for why it is necessary to keep reference of object:
                          https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/1824
        :param kwargs: keyword arguments e.g. il=['input1.tif', App_object2, App_object3.out], out='output.tif'
        """
        self.appname = appname
        self.app = otbApplication.Registry.CreateApplication(appname)
        self.output_parameters_keys = self.get_output_parameters_keys()
        self.set_parameters(*args, **kwargs)
        if execute:
            self.app.Execute()
        if image_dic:
            self.image_dic = image_dic

        # 'Saving' outputs as attributes, i.e. so that they can be accessed like that: App.out
        # Also, thanks to __getitem__ method, the outputs can be accessed as App["out"]. This is useful when the key
        # contains reserved characters such as a point eg "io.out"
        for output_param_key in self.output_parameters_keys:
            output = Output(self.app, output_param_key)
            setattr(self, output_param_key, output)

        # Writing outputs to disk if needed
        if any([output_param_key in kwargs for output_param_key in self.output_parameters_keys]):
            self.app.ExecuteAndWriteOutput()

    def set_parameters(self, *args, **kwargs):
        """
        Set some parameters of the app. When useful, e.g. for images list, this function appends the parameters instead
        of overwriting them. Handles any parameters, i.e. in-memory & filepaths
        :param args: Can be : - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                                (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                              - string, App or Output, useful when the user implicitly wants to set the param "in"
                              - list, useful when the user implicitly wants to set the param "il"
        :param kwargs: keyword arguments e.g. il=['input1.tif', oApp_object2, App_object3.out], out='output.tif'
        :return:
        """
        if args:
            for arg in args:
                if isinstance(arg, dict):
                    kwargs.update(arg)
                elif isinstance(arg, (str, App, Output, Input, Operation)):
                    kwargs.update({'in': arg})
                elif isinstance(arg, list):
                    kwargs.update({'il': arg})

        # Going through all arguments
        for k, v in kwargs.items():
            if isinstance(v, App):
                self.app.ConnectImage(k, v.app, v.output_parameters_keys[0])
            elif isinstance(v, (Output, Input, Operation)):
                self.app.ConnectImage(k, v.app, v.output_parameter_key)
            elif isinstance(v, otbApplication.Application):
                outparamkey = [param for param in v.GetParametersKeys()
                               if v.GetParameterType(param) == otbApplication.ParameterType_OutputImage][0]
                self.app.ConnectImage(k, v, outparamkey)
            elif k == 'ram':  # SetParameterValue in OTB<7.4 doesn't work for ram parameter cf gitlab OTB issue 2200
                self.app.SetParameterInt('ram', int(v))
            elif not isinstance(v, list):  # any other parameters (str, int...)
                self.app.SetParameterValue(k, v)  # we use SetParameterValue
            elif isinstance(v, list):
                other_types = []  # any other parameters (str, int...) whose key is not "il"
                # To enable in-memory connections, we go through the list and set the parameters of the list one by one.
                for input in v:
                    if isinstance(input, App):
                        self.app.ConnectImage(k, input.app, input.output_parameters_keys[0])
                    elif isinstance(input, (Output, Input, Operation)):
                        self.app.ConnectImage(k, input.app, input.output_parameter_key)
                    elif isinstance(input, otbApplication.Application):
                        outparamkey = [param for param in input.GetParametersKeys()
                                       if input.GetParameterType(param) == otbApplication.ParameterType_OutputImage][0]
                        self.app.ConnectImage(k, input, outparamkey)
                    elif k == 'il':  # specific case so that we do not overwrite any previously set element of 'il'
                        # TODO: this is ugly, to be fixed
                        self.app.AddParameterStringList(k, input)
                    else:
                        other_types.append(input)
                if len(other_types) > 0:
                    self.app.SetParameterValue(k, other_types)

    def get_output_parameters_keys(self):
        """
        :return: list of output parameters keys, e.g ['out']
        """
        output_param_keys = [param for param in self.app.GetParametersKeys()
                             if self.app.GetParameterType(param) == otbApplication.ParameterType_OutputImage]
        return output_param_keys

    def __str__(self):
        return '<pyotb.App {} object id {}>'.format(self.appname, id(self))


# CONSTANTS
# We run this piece of code inside a independent `multiprocessing.Process` because of a current (2021-11) bug that
# prevents the use of OTBTF and tensorflow inside the same script
def get_available_applications(q):
    import otbApplication
    q.put(otbApplication.Registry.GetAvailableApplications())


q = multiprocessing.Queue()
p = multiprocessing.Process(target=get_available_applications, args=(q,))
p.start()
p.join()
AVAILABLE_APPLICATIONS = q.get(block=False)

# This is to enable aliases of Apps, i.e. using apps like `pyotb.AppName(...)` instead of `pyotb.App('AppName', ...)`
if AVAILABLE_APPLICATIONS:
    for app_name in AVAILABLE_APPLICATIONS:
        exec(f"""def {app_name}(*args, **kwargs): return App('{app_name}', *args, **kwargs)""")


class Operation(otbObject):
    """Class for all arithmetic operations.

    Example
    -------
    Consider the python expression (input1 + 2 * input2)  >  0.
    This class enables to create a BandMathX app, with expression such as (im2 + 2 * im1) > 0 ? 1 : 0

    The order in which the expression is executed is determined by Python.
    (input1 + 2 * input2)  >  0
             |__________|
                  |
            Operation1, with expression 2 * im1
    |__________________|
                |
          Operation2, with expression im2 + 2 * im1
    |__________________________|
                        |
                  Operation3, with expression (im2 + 2 * im1) > 0 ? 1 : 0

    """
    def __init__(self, operator, input1, input2=None):
        """
        Given an operation involving 1 or 2 inputs, this function handles the naming of inputs (such as im1, im2).

        :param operator: one of +, -, *, /, >, <, >=, <=, &, |, abs.
        :param input1: first input. Can be App, Output, Input, Operation, str (filepath), int or float
        :param input2: second input. Optional. Can be App, Output, Input, Operation, str (filepath), int or float
        """
        self.operator = operator
        self.input1 = input1
        self.input2 = input2

        # We first create a 'fake' expression. E.g for the operation +, we create a fake expression that is like
        # str(input1) + str(input2)

        inputs = []
        nb_channels = {}
        # We begin with potential Operation objects and save their attributes
        if isinstance(input1, Operation):
            fake_exp1 = input1.fake_exp
            inputs.extend(input1.inputs)
            nb_channels.update(input1.nb_channels)
        # For int or float input, we just need to save their value
        elif isinstance(input1, (int, float)):
            fake_exp1 = str(input1)
        # We go on with "regular input", i.e. pyotb objects, filepaths...
        else:
            nb_channels[input1] = get_nbchannels(input1)
            inputs.append(input1)
            fake_exp1 = str(input1)

        if input2 is None:
            fake_exp = f'({operator}({fake_exp1}))'
        else:
            # We begin with potential Operation objects and save their attributes
            if isinstance(input2, Operation):
                fake_exp2 = input2.fake_exp
                inputs.extend(input2.inputs)
                nb_channels.update(input2.nb_channels)
            # We go on with "regular input", i.e. App, filepaths...
            elif not isinstance(input2, (Operation, int, float)):
                nb_channels[input2] = get_nbchannels(input2)
                inputs.append(input2)
                fake_exp2 = str(input2)
            # For int or float input, we just need to save their value
            elif isinstance(input2, (int, float)):
                fake_exp2 = str(input2)

            # We create here the "fake" expression. For example, for a BandMathX expression such as '2 * im1 + im2',
            # the false expression stores the expression 2 * str(input1) + str(input2)
            if operator in ['>', '<', '>=', '<=', '==', '!=']:
                fake_exp = f'({fake_exp1} {operator} {fake_exp2} ? 1 : 0)'
            else:
                fake_exp = f'({fake_exp1} {operator} {fake_exp2})'

        self.fake_exp, self.inputs, self.nb_channels = fake_exp, inputs, nb_channels

        # creating a dictionary that is like {str(input1): 'im1', '/tmp/image.tif': 'im2', ...}.
        # NB: the keys of the dictionary are strings-only, instead of 'complex' objects, to enable easy serialization
        self.im_dic = {}
        im_count = 1
        mapping_str_to_input = {}  # to be able to retrieve the real python object from its string representation
        for input in self.inputs:
            if not isinstance(input, (int, float)):
                if str(input) not in self.im_dic:
                    self.im_dic[str(input)] = 'im{}'.format(im_count)
                    mapping_str_to_input[str(input)] = input
                    im_count += 1

        print(self.im_dic, self.nb_channels, self.inputs)  # TODO: just for debug, to be removed

        # getting unique image inputs, in the order im1, im2, im3 ...
        self.unique_inputs = [mapping_str_to_input[str_input] for str_input in sorted(self.im_dic, key=self.im_dic.get)]
        self.output_parameter_key = 'out'

        # Computing the bmx app
        bmx = App('BandMathX', il=self.unique_inputs, exp=self.get_real_exp())
        self.app = bmx.app

    def get_real_exp(self):
        """Generates the BandMathX expression"""
        # Checking that all images have the same number of channels
        if any([value != next(iter(self.nb_channels.values())) for value in self.nb_channels.values()]):
            raise Exception('All images do not have the same number of bands')

        exp = ''
        for band in range(1, next(iter(self.nb_channels.values())) + 1):
            one_band_exp = self.fake_exp
            # Computing the expression
            for input in self.inputs:
                # replace the name of in-memory object (e.g. '<pyotb.App object, id 139912042337952>' by 'im1b1')
                one_band_exp = one_band_exp.replace(str(input), self.im_dic[str(input)] + f'b{band}')

            exp += one_band_exp
            # concatenate bands
            if band < next(iter(self.nb_channels.values())) and next(iter(self.nb_channels.values())) > 1:
                exp += ';'
        return exp

    def __str__(self):
        if self.input2 is not None:
            return '<pyotb.Operation object, {} {} {}, id {}>'.format(str(self.input1), self.operator, str(self.input2),
                                                                      id(self))
        else:
            return '<pyotb.Operation object, {} {}, id {}>'.format(self.operator, str(self.input1), id(self))


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
        logging.info('The condition has one channel whereas X/Y has/have {} channels. Expanding number of channels '
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
        logging.warning('All images may not have the same CRS, which might cause unpredictable results')

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

        logging.info(
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
                logging.error(e)
                logging.error('Images may not intersect : {}'.format(input))
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

        logging.info('Resampling all inputs to resolution : {}'.format(metadatas[reference_input]['GeoTransform'][1]))

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
            model_serve.set_parameters({'source{}.il'.format(i+1): [input]})

        model_serve.Execute()
        # TODO: handle the deletion of the temporary model ?

        return model_serve
    return wrapper


def get_nbchannels(inp):
    """
    Get the encoding of input image pixels
    :param inp: a parameterimage or str
    """
    # Executing the app, without printing its log
    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        info = App("ReadImageInfo", inp)
        sys.stdout = stdout
        nb_channels = info.GetParameterInt("numberbands")
    except Exception as e:
        sys.stdout = stdout
        logging.error('Not a valid image : {}'.format(inp))
        logging.error(e)
        nb_channels = None
    sys.stdout = stdout
    return nb_channels


def get_pixel_type(inp):
    """
    Get the encoding of input image pixels
    :param inp: a filepath, or any pyotb object
    :return pixel_type: either a dict of pixel types (in case of an App where there are several outputs)
                        format is like `otbApplication.ImagePixelType_uint8'
    """
    if isinstance(inp, str):
        # Executing the app, without printing its log
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        info = App("ReadImageInfo", inp)
        sys.stdout = stdout
        datatype = info.GetParameterInt("datatype")  # which is such as short, float...
        dataype_to_pixeltype = {'unsigned_char': 'uint8', 'short': 'int16', 'unsigned_short': 'uint16', 'int': 'int32',
                                'unsigned_int': 'uint32', 'long': 'int32', 'ulong': 'uint32', 'float': 'float',
                                'double': 'double'}
        pixel_type = dataype_to_pixeltype[datatype]
        pixel_type = getattr(otbApplication, 'ImagePixelType_{}'.format(pixel_type))
    elif isinstance(inp, (Input, Output, Operation)):
        pixel_type = inp.GetImageBasePixelType(inp.output_parameter_key)
    elif isinstance(inp, App):
        if len(inp.output_parameters_keys) > 1:
            pixel_type = inp.GetImageBasePixelType(inp.output_parameters_keys[0])
        else:
            pixel_type = {key: inp.GetImageBasePixelType(key) for key in inp.output_parameters_keys}

    return pixel_type


if __name__ == '__main__':
    # Juste an example of Resampling + ROI extraction + Band extraction
    #                     + Smoothing + Subtraction + Thresholding to form binary mask
    input_path = '/mnt/mo-gpu/decloud/HAL_notmp/S2_PREPARE/T31TCJ/SENTINEL2B_20200803-105901-761_L2A_T31TCJ_C_V2-2/' \
                 'SENTINEL2B_20200803-105901-761_L2A_T31TCJ_C_V2-2_FRE_10m.tif'

    # ============
    # With pyotb
    # ============
    resampled = App('RigidTransformResample', {'in': input_path, 'interpolator': 'linear',
                                               'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})
    first_band_extracted = resampled[:1000, :1000, 0]
    smoothed = App('Smoothing', first_band_extracted, {'type.mean.radius': 6}, type='mean')
    diff_thresholded = abs(first_band_extracted + 3 - smoothed) > 10
    diff_thresholded.write('/tmp/diff_thresholded_with_wrapper.tif')

    # =========================
    # Equivalent with only OTB
    # =========================
    resampled = otbApplication.Registry.CreateApplication('RigidTransformResample')
    resampled.SetParameterString('in', input_path)
    resampled.SetParameterString('interpolator', 'linear')
    resampled.SetParameterFloat('transform.type.id.scalex', 0.5)
    resampled.SetParameterFloat('transform.type.id.scaley', 0.5)
    resampled.Execute()

    extracted = otbApplication.Registry.CreateApplication('ExtractROI')
    extracted.ConnectImage('in', resampled, 'out')
    extracted.SetParameterString('mode', 'extent')
    extracted.SetParameterString('mode.extent.unit', 'pxl')
    extracted.SetParameterFloat('mode.extent.ulx', 0)
    extracted.SetParameterFloat('mode.extent.uly', 0)
    extracted.SetParameterFloat('mode.extent.lrx', 999)
    extracted.SetParameterFloat('mode.extent.lry', 999)
    extracted.Execute()  # otherwise channel extraction doesn't work with in-memory image
    extracted.SetParameterStringList('cl', ['Channel1'])
    extracted.Execute()

    smoothed = otbApplication.Registry.CreateApplication('Smoothing')
    smoothed.ConnectImage('in', extracted, 'out')
    smoothed.SetParameterString("type", 'mean')
    smoothed.SetParameterInt("type.mean.radius", 6)
    smoothed.Execute()

    bmx = otbApplication.Registry.CreateApplication('BandMathX')
    bmx.ConnectImage('il', extracted, 'out')
    bmx.ConnectImage('il', smoothed, 'out')
    bmx.SetParameterString('exp', "abs(im1b1 +3 - im2b1) > 10 ? 1 : 0")
    bmx.SetParameterString('out', '/tmp/diff_thresholded.tif')
    bmx.ExecuteAndWriteOutput()

    # In pyotb, writing intermediate results is possible for debug :
    # resampled.write('/tmp/resampled.tif', is_intermediate=True)
    # smoothed.write('/tmp/smoothed.tif', is_intermediate=True)
