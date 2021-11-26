import otbApplication
import logging
import sys
import os
from abc import ABC
import numpy as np

logger = logging
logger.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
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
                    logger.warning('Keyword arguments specified, ignoring argument: {}'.format(arg))
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
                    logger.info("Using extended filename for output.")
                logger.info("Write output for \"{}\" in {}".format(output_parameter_key, out_fn))
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

        # Writing outputs to disk if needed
        if any([output_param_key in kwargs for output_param_key in self.output_parameters_keys]):
            self.app.ExecuteAndWriteOutput()


    def get_output_parameters_keys(self):
        """
        :return: list of output parameters keys, e.g ['out']
        """
        output_param_keys = [param for param in self.app.GetParametersKeys()
                             if self.app.GetParameterType(param) == otbApplication.ParameterType_OutputImage]
        return output_param_keys

    def __str__(self):
        return '<pyotb.App {} object id {}>'.format(self.appname, id(self))


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
            # For int or float input, we just need to save their value
            elif isinstance(input2, (int, float)):
                fake_exp2 = str(input2)
            # We go on with "regular input", i.e. pyotb objects, filepaths...
            else:
                nb_channels[input2] = get_nbchannels(input2)
                inputs.append(input2)
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
        logger.error('Not a valid image : {}'.format(inp))
        logger.error(e)
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
