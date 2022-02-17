# -*- coding: utf-8 -*-
import logging
from abc import ABC
from pathlib import Path

import numpy as np
import otbApplication as otb

logger = logging.getLogger()


class otbObject(ABC):
    """
    Abstract class that gathers common operations for any OTB in-memory raster.
    All child of this class must have an `app` attribute that is an OTB application.
    """
    @property
    def shape(self):
        """
        Enables to retrieve the shape of a pyotb object. Can not be called before app.Execute()
        :return shape: (width, height, bands)
        """
        if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation, Slicer
            output_parameter_key = self.output_parameter_key
        else:  # this is for App
            output_parameter_key = self.output_parameters_keys[0]
        image_size = self.GetImageSize(output_parameter_key)
        image_bands = self.GetImageNbBands(output_parameter_key)
        # TODO: it currently returns (width, height, bands), should we use numpy convention (height, width, bands) ?
        return (*image_size, image_bands)

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
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, str) and kwargs:
                logger.warning(f'{self.name}: Keyword arguments specified, ignoring argument "{arg}"')
            elif isinstance(arg, str):
                if hasattr(self, 'output_parameter_key'):  # this is for Output, Operation...
                    output_parameter_key = self.output_parameter_key
                else:  # this is for App
                    output_parameter_key = self.output_parameters_keys[0]
                kwargs.update({output_parameter_key: arg})

        # Handling pixel types
        pixel_types = {}
        if isinstance(pixel_type, str):  # this correspond to 'uint8' etc...
            pixel_type = getattr(otb, f'ImagePixelType_{pixel_type}')
            pixel_types = {param_key: pixel_type for param_key in kwargs}
        elif isinstance(pixel_type, int):  # this corresponds to ImagePixelType_...
            pixel_types = {param_key: pixel_type for param_key in kwargs}
        elif isinstance(pixel_type, dict):  # this is to specify a different pixel type for each output
            for key, this_pixel_type in pixel_type.items():
                if isinstance(this_pixel_type, str):
                    this_pixel_type = getattr(otb, f'ImagePixelType_{this_pixel_type}')
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
                    logger.info(f'{self.name}: Using extended filename for output.')
                logger.info(f'{self.name}: write output file "{output_parameter_key}" to {out_fn}')
                self.app.SetParameterString(output_parameter_key, out_fn)

                if output_parameter_key in pixel_types:
                    self.app.SetParameterOutputImagePixelType(output_parameter_key, pixel_types[output_parameter_key])

            self.app.ExecuteAndWriteOutput()
            self.app.Execute()  # this is just to be able to use the object in in-memory pipelines without problems

    # Special methods
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
        return Slicer(self, rows, cols, channels)

    def __getattr__(self, name):
        """
        This method is called when the default attribute access fails. We choose to try to access the attribute of
        self.app. Thus, any method of otbApplication can be used transparently on otbObject objects,
        e.g. SetParameterOutputImagePixelType() or ExportImage() work
        """
        return getattr(self.app, name)

    def __add__(self, other):
        """Overrides the default addition and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', self, other)

    def __sub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', self, other)

    def __mul__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', self, other)

    def __truediv__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', self, other)

    def __radd__(self, other):
        """Overrides the default reverse addition and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', other, self)

    def __rsub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', other, self)

    def __rmul__(self, other):
        """Overrides the default multiplication and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', other, self)

    def __rtruediv__(self, other):
        """Overrides the default division and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', other, self)

    def __abs__(self):
        """Overrides the default abs operator and flavours it with BandMathX"""
        return Operation('abs', self)

    def __ge__(self, other):
        """Overrides the default greater or equal and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>=', self, other)

    def __le__(self, other):
        """Overrides the default less or equal and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<=', self, other)

    def __gt__(self, other):
        """Overrides the default greater operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>', self, other)

    def __lt__(self, other):
        """Overrides the default less operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<', self, other)

    def __eq__(self, other):
        """Overrides the default eq operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('==', self, other)

    def __ne__(self, other):
        """Overrides the default different operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('!=', self, other)

    def __or__(self, other):
        """Overrides the default or operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('||', self, other)

    def __and__(self, other):
        """Overrides the default and operator and flavours it with BandMathX"""
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('&&', self, other)

    # TODO: other operations ?
    #  e.g. __pow__... cf https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __hash__(self):
        return id(self)

    def __array__(self):
        """
        This is called when running np.asarray(pyotb_object)
        :return: a numpy array
        """
        if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation, Slicer
            output_parameter_key = self.output_parameter_key
        else:  # this is for App
            output_parameter_key = self.output_parameters_keys[0]
        return self.app.ExportImage(output_parameter_key)['array']

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        This is called whenever a numpy function is called on a pyotb object.
        :param ufunc: numpy function
        :param method: a internal numpy argument
        :param inputs: inputs, at least one being pyotb object. If there are several pyotb objects, they must all have
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
                elif isinstance(input, (App, Input, Output, Operation, Slicer)):
                    if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation, Slicer
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
            app = App('ExtractROI', image_dic=result_dic, execute=False)  # pass the result_dic just to keep reference
            if result_array.shape[2] == 1:
                app.ImportImage('in', result_dic)
            else:
                app.ImportVectorImage('in', result_dic)
            app.Execute()
            return app

        else:
            return NotImplemented


class Slicer(otbObject):
    """Slicer objects i.e. when we call something like raster[:, :, 2] from Python"""

    def __init__(self, input, rows, cols, channels):
        """
        Create a slicer object, that can be used directly for writing or inside a BandMath :
        - an ExtractROI app that handles extracting bands and ROI and can be written to disk or used in pipelines
        - in case the user only wants to extract one band, an expression such as "im1b#"
        :param input:
        :param rows:
        :param cols:
        :param channels:
        """
        # Initialize the app that will be used for writing the slicer
        app = App('ExtractROI', {"in": input, 'mode': 'extent'})
        self.output_parameter_key = 'out'
        self.name = 'Slicer'

        # Channel slicing
        nb_channels = get_nbchannels(input)
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
            elif isinstance(channels, tuple):
                channels = list(channels)
            elif not isinstance(channels, list):
                raise ValueError(f'Invalid type for channels, should be int, slice or list of bands. : {channels}')

            # Change the potential negative index values to reverse index
            channels = [c if c >= 0 else nb_channels + c for c in channels]

            app.set_parameters(cl=[f'Channel{i + 1}' for i in channels])

        # Spatial slicing
        spatial_slicing = False
        # TODO: handle PixelValue app so that accessing value is possible, e.g. raster[120, 200, 0]
        # TODO TBD: handle the step value in the slice so that NN undersampling is possible ? e.g. obj[::2, ::2]
        if rows.start is not None:
            app.set_parameters({'mode.extent.uly': rows.start})
            spatial_slicing = True
        if rows.stop is not None and rows.stop != -1:
            app.set_parameters(
                {'mode.extent.lry': rows.stop - 1})  # subtract 1 to be compliant with python convention
            spatial_slicing = True
        if cols.start is not None:
            app.set_parameters({'mode.extent.ulx': cols.start})
            spatial_slicing = True
        if cols.stop is not None and cols.stop != -1:
            app.set_parameters(
                {'mode.extent.lrx': cols.stop - 1})  # subtract 1 to be compliant with python convention
            spatial_slicing = True

        # keeping the OTB app, not the pyotb app
        self.app = app.app
        self.app.Execute()

        # These are some attributes when the user simply wants to extract *one* band to be used in an Operation
        if not spatial_slicing and isinstance(channels, list) and len(channels) == 1:
            self.one_band_sliced = channels[0] + 1  # OTB convention: channels start at 1
            self.input = input


class Input(otbObject):
    """
    Class for transforming a filepath to pyOTB object
    """
    def __init__(self, filepath):
        self.app = App('ExtractROI', filepath).app
        self.output_parameter_key = 'out'
        self.filepath = filepath
        self.name = f'Input from {filepath}'

    def __str__(self):
        return '<pyotb.Input object from {}>'.format(self.filepath)


class Output(otbObject):
    """
    Class for output of an app
    """
    def __init__(self, app, output_parameter_key):
        self.app = app  # keeping a reference of the OTB app
        self.output_parameter_key = output_parameter_key
        self.name = f'Output {output_parameter_key} from {self.app.GetName()}'

    def __str__(self):
        return '<pyotb.Output {} object, id {}>'.format(self.app.GetName(), id(self))



class App(otbObject):
    """
    Class of an OTB app
    """
    _name = ""
    @property
    def name(self):
        """Property to store a custom app name that will be displayed in logs"""
        return self._name or self.appname

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def finished(self):
        """Property to store whether or not App has been executed"""
        if self._finished and self.find_output():
            return True
        return False

    @finished.setter
    def finished(self, val):
        # This will only store if app has been excuted, then find_output() is called when accessing the property
        self._finished = val

    def __init__(self, appname, *args, execute=True, image_dic=None, otb_stdout=True, **kwargs):
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
        if otb_stdout:
            self.app = otb.Registry.CreateApplication(appname)
        else:
            self.app = otb.Registry.CreateApplicationWithoutLogger(appname)
        self.image_dic = image_dic
        self._finished = False
        # Parameters
        self.parameters = {}
        self.output_parameters_keys = self.get_output_parameters_keys()
        if args or kwargs:
            self.set_parameters(*args, **kwargs)
        else:
            logger.warning(f"{self.name}: No parameters where provided. Use App.set_parameters() then App.execute()")
            execute = False
        # Run app, write output if needed, update `finished` property
        if execute:
            self.execute()
        # 'Saving' outputs as attributes, i.e. so that they can be accessed like that: App.out
        # Also, thanks to __getitem__ method, the outputs can be accessed as App["out"]. This is useful when the key
        # contains reserved characters such as a point eg "io.out"
        for key in self.output_parameters_keys:
            output = Output(self.app, key)
            setattr(self, key, output)

    def execute(self):
        """
        Execute with appropriate App method and outputs to disk if needed"
        :return: boolean flag that indicate if command executed with success
        """
        logger.debug(f"{self.name}: run execute() with parameters={self.parameters}")
        try:
            if self.__with_output():
                self.app.ExecuteAndWriteOutput()
            self.app.Execute()
            self.finished = True
        except (RuntimeError, FileNotFoundError) as e:
            raise Exception(f'{self.name}: error during during app execution') from e
        logger.debug(f"{self.name}: execution succeeded")
        return self.finished

    def find_output(self):
        """
        Find output files in directory using parameters
        :return: list of files found on disk
        """
        if not self.__with_output():
            return
        files = []
        missing = []
        for param in self.output_parameters_keys:
            filename = self.parameters[param]
            if Path(filename).exists():
                files.append(filename)
            else:
                missing.append(filename)
        if missing:
            for filename in missing:
                logger.error(f"{self.name}: execution seems to have failed, {filename} does not exist")
                #raise FileNotFoundError(filename)
        return files

    def clear(self, parameters=True, memory=False):
        """
        Free ressources and reset App state
        :param parameters: to clear settings dictionnary
        :param memory: to free app ressources in memory
        """
        if parameters:
            for p in self.parameters:
                self.app.ClearValue(p)
        if memory:
            self.app.FreeRessources()

    def get_output_parameters_keys(self):
        return [param for param in self.app.GetParametersKeys()
                if self.app.GetParameterType(param) == otb.ParameterType_OutputImage]

    def set_parameters(self, *args, **kwargs):
        """
        Set some parameters of the app. When useful, e.g. for images list, this function appends the parameters instead
        of overwriting them. Handles any parameters, i.e. in-memory & filepaths
        :param args: Can be : - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                                (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                              - string, App or Output, useful when the user implicitly wants to set the param "in"
                              - list, useful when the user implicitly wants to set the param "il"
        :param kwargs: keyword arguments e.g. il=['input1.tif', oApp_object2, App_object3.out], out='output.tif'
        :return: boolean flag that indicate if app was correctly set using given parameters
        """
        self.parameters.update(self.__parse_args(args))
        self.parameters.update(kwargs)
        # Going through all arguments
        for param, obj in self.parameters.items():
            if param not in self.app.GetParametersKeys():
                raise Exception(f"{self.name}: parameter '{param}' was not recognized. "
                                f"Available keys are {self.app.GetParametersKeys()}")
            # When the parameter expects a list, if needed, change the value to list
            if self.__is_key_list(param) and not isinstance(obj, (list, tuple)):
                self.parameters[param] = [obj]
                obj = [obj]
            try:
                # This is when we actually call self.app.SetParameter*
                self.__set_param(param, obj)
            except (RuntimeError, TypeError, ValueError) as e:
                raise Exception(f"{self.name}: something went wrong before execution "
                                f"(while setting parameter {param} to '{obj}')") from e

    # Private functions
    @staticmethod
    def __parse_args(args):
        """Gather all input arguments in kwargs dict"""
        kwargs = {}
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, (str, App, Output, Input, Operation, Slicer)):
                kwargs.update({'in': arg})
            elif isinstance(arg, list):
                kwargs.update({'il': arg})
        return kwargs

    def __set_param(self, param, obj):
        """Decide which otb.Application method to use depending on target object"""
        # Single-parameter cases
        if isinstance(obj, App):
            self.app.ConnectImage(param, obj.app, obj.output_parameters_keys[0])
        elif isinstance(obj, (Output, Input, Operation)):
            self.app.ConnectImage(param, obj.app, obj.output_parameter_key)
        elif isinstance(obj, otb.Application):  # this is for backward comp with plain OTB
            outparamkey = [param for param in obj.GetParametersKeys()
                            if obj.GetParameterType(param) == otb.ParameterType_OutputImage][0]
            self.app.ConnectImage(param, obj, outparamkey)
        elif param == 'ram':  # SetParameterValue in OTB<7.4 doesn't work for ram parameter cf gitlab OTB issue 2200
            self.app.SetParameterInt('ram', int(obj))
        elif not isinstance(obj, list):  # any other parameters (str, int...)
            self.app.SetParameterValue(param, obj)
        ### Parameter list cases
        # Images list
        elif self.__is_key_images_list(param):
            # To enable possible in-memory connections, we go through the list and set the parameters one by one
            for input in obj:
                if isinstance(input, App):
                    self.app.ConnectImage(param, input.app, input.output_parameters_keys[0])
                elif isinstance(input, (Output, Input, Operation, Slicer)):
                    self.app.ConnectImage(param, input.app, input.output_parameter_key)
                elif isinstance(input, otb.Application):  # this is for backward comp with plain OTB
                    outparamkey = [param for param in input.GetParametersKeys() if
                                    input.GetParameterType(param) == otb.ParameterType_OutputImage][0]
                    self.app.ConnectImage(param, input, outparamkey)
                else:  # here `input` should be an image filepath
                    # Append `input` to the list, do not overwrite any previously set element of the image list
                    self.app.AddParameterStringList(param, input)
        # List of any other types (str, int...)
        else:
            self.app.SetParameterValue(param, obj)


    def __with_output(self):
        """Check if App has any output parameter key"""
        return any([k in self.parameters for k in self.output_parameters_keys])

    def __is_key_list(self, key):
        """Check if a key of the App is an input parameter list"""
        return self.app.GetParameterType(key) in (
            otb.ParameterType_InputImageList,
            otb.ParameterType_StringList,
            otb.ParameterType_InputFilenameList,
            otb.ParameterType_InputVectorDataList,
            otb.ParameterType_ListView
        )

    def __is_key_images_list(self, key):
        """Check if a key of the App is an input parameter image list"""
        return self.app.GetParameterType(key) in (
            otb.ParameterType_InputImageList,
            otb.ParameterType_InputFilenameList
        )

    # Special methods
    def __str__(self):
        return f'<pyotb.App {self.appname} object id {id(self)}>'


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

    def __init__(self, operator, *inputs, nb_bands=None):
        """
        Given some inputs and an operator, this function enables to transform this into an OTB application.
        Operations generally involve 2 inputs (+, -...). It can have only 1 input for `abs` operator.
        It can have 3 inputs for the ternary operator `cond ? x : y`,

        :param operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
        :param inputs: inputs. Can be App, Output, Input, Operation, Slicer, filepath, int or float
        :param nb_bands: to specify the output nb of bands. Optional
        """
        self.operator = operator

        # We first create a 'fake' expression. E.g for the operation `input1 + input2` , we create a fake expression
        # that is like "str(input1) + str(input2)"
        self.create_fake_exp(operator, inputs, nb_bands=nb_bands)

        # Transforming images to the adequate im#, e.g. `input1` to "im1"
        # creating a dictionary that is like {str(input1): 'im1', 'image2.tif': 'im2', ...}.
        # NB: the keys of the dictionary are strings-only, instead of 'complex' objects, to enable easy serialization
        self.im_dic = {}
        self.im_count = 1
        mapping_str_to_input = {}  # to be able to retrieve the real python object from its string representation
        for input in self.inputs:
            if not isinstance(input, (int, float)):
                if str(input) not in self.im_dic:
                    self.im_dic[str(input)] = 'im{}'.format(self.im_count)
                    mapping_str_to_input[str(input)] = input
                    self.im_count += 1

        # getting unique image inputs, in the order im1, im2, im3 ...
        self.unique_inputs = [mapping_str_to_input[str_input] for str_input in sorted(self.im_dic, key=self.im_dic.get)]
        self.output_parameter_key = 'out'

        # Computing the BandMath or BandMathX app
        self.exp_bands, self.exp = self.get_real_exp(self.fake_exp_bands)
        if len(self.exp_bands) == 1:
            self.app = App('BandMath', il=self.unique_inputs, exp=self.exp).app
        else:
            self.app = App('BandMathX', il=self.unique_inputs, exp=self.exp).app

        self.name = f'Operation exp="{self.exp}"'

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        """
        We first create a 'fake' expression. E.g for the operation input1 + input2 , we create a fake expression
        that is like "str(input1) + str(input2)"

        :param operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
        :param inputs: inputs. Can be App, Output, Input, Operation, Slicer, filepath, int or float
        :param nb_bands: to specify the output nb of bands. Optional
        """
        self.inputs = []
        self.nb_channels = {}

        logger.debug(f"{operator}, {inputs}")
        # this is when we use the ternary operator with `pyotb.where` function. The output nb of bands is already known
        if operator == '?' and nb_bands:
            nb_bands = nb_bands
        # For any other operations, the output number of bands is the same as inputs
        else:
            if any([isinstance(input, Slicer) and hasattr(input, 'one_band_sliced') for input in inputs]):
                nb_bands = 1
            else:
                nb_bands_list = [get_nbchannels(input) for input in inputs if not isinstance(input, (float, int))]
                # check that all inputs have the same nb of bands
                if len(nb_bands_list) > 1:
                    if not all(x == nb_bands_list[0] for x in nb_bands_list):
                        raise Exception('All images do not have the same number of bands')
                nb_bands = nb_bands_list[0]

        # Create a list of fake expressions, each item of the list corresponding to one band
        self.fake_exp_bands = []
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for k, input in enumerate(inputs):
                # Generating the fake expression of the current input
                # this is a special case for the condition of the ternary operator `cond ? x : y`
                if len(inputs) == 3 and k == 0:
                    # when cond is monoband whereas the result is multiband, we expand the cond to multiband
                    if nb_bands != input.shape[2]:
                        cond_band = 1
                    else:
                        cond_band = band
                    fake_exp, corresponding_inputs, nb_channels = self._create_one_input_fake_exp(input, cond_band,
                                                                                                  keep_logical=True)
                # any other input
                else:
                    fake_exp, corresponding_inputs, nb_channels = self._create_one_input_fake_exp(input, band,
                                                                                                  keep_logical=False)

                fake_exps.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresponding_inputs and nb_channels:
                    self.inputs.extend(corresponding_inputs)
                    self.nb_channels.update(nb_channels)

            # Generating the fake expression of the whole operation
            if len(inputs) == 1:  # this is only for 'abs'
                fake_exp = f'({operator}({fake_exps[0]}))'
            elif len(inputs) == 2:
                # We create here the "fake" expression. For example, for a BandMathX expression such as '2 * im1 + im2',
                # the false expression stores the expression 2 * str(input1) + str(input2)
                fake_exp = f'({fake_exps[0]} {operator} {fake_exps[1]})'
            elif len(inputs) == 3 and operator == '?':  # this is only for ternary expression
                fake_exp = f'({fake_exps[0]} ? {fake_exps[1]} : {fake_exps[2]})'

            self.fake_exp_bands.append(fake_exp)

    def _create_one_input_fake_exp(self, input, band, keep_logical=False):
        """
        This an internal function, only to be used by `create_fake_exp`. Enable to create a fake expression just for one
        input and one band.
        :param input:
        :param band: which band to consider (bands start at 1)
        :param keep_logical: whether to keep the logical expressions "as is" in case the input is a logical operation.
                    ex: if True, for `input1 > input2`, returned fake expression is "str(input1) > str(input2)"
                        if False, for `input1 > input2`, returned fake expression is "str(input1) > str(input2) ? 1 : 0"
        """
        # Special case for one-band slicer
        if isinstance(input, Slicer) and hasattr(input, 'one_band_sliced'):
            if keep_logical and isinstance(input.input, logicalOperation):
                fake_exp = input.input.logical_fake_exp_bands[input.one_band_sliced - 1]
                inputs = input.input.inputs
                nb_channels = input.input.nb_channels
            elif isinstance(input.input, Operation):
                # keep only one band of the expression
                fake_exp = input.input.fake_exp_bands[input.one_band_sliced - 1]
                inputs = input.input.inputs
                nb_channels = input.input.nb_channels
            else:
                # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
                fake_exp = str(input.input) + f'b{input.one_band_sliced}'
                inputs = [input.input]
                nb_channels = {input.input: 1}
        # For logicalOperation, we save almost the same attributes as an Operation
        elif keep_logical and isinstance(input, logicalOperation):
            fake_exp = input.logical_fake_exp_bands[band - 1]
            inputs = input.inputs
            nb_channels = input.nb_channels
        elif isinstance(input, Operation):
            fake_exp = input.fake_exp_bands[band - 1]
            inputs = input.inputs
            nb_channels = input.nb_channels
        # For int or float input, we just need to save their value
        elif isinstance(input, (int, float)):
            fake_exp = str(input)
            inputs = None
            nb_channels = None
        # We go on with other inputs, i.e. pyotb objects, filepaths...
        else:
            nb_channels = {input: get_nbchannels(input)}
            inputs = [input]
            # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
            fake_exp = str(input) + f'b{band}'

        return fake_exp, inputs, nb_channels

    def get_real_exp(self, fake_exp_bands):
        """Generates the BandMathX expression"""
        # Create a list of expression, each item corresponding to one band (e.g. ['im1b1 + 1', 'im1b2 + 1'])
        exp_bands = []
        for one_band_fake_exp in fake_exp_bands:
            one_band_exp = one_band_fake_exp
            for input in self.inputs:
                # replace the name of in-memory object (e.g. '<pyotb.App object>b1' by 'im1b1')
                one_band_exp = one_band_exp.replace(str(input), self.im_dic[str(input)])
            exp_bands.append(one_band_exp)

        # Form the final expression (e.g. 'im1b1 + 1; im1b2 + 1')
        exp = ';'.join(exp_bands)

        return exp_bands, exp

    def __str__(self):
        return f'<pyotb.Operation `{self.operator}` object, id {id(self)}>'


class logicalOperation(Operation):
    """
    This is a specialization of Operation class for boolean logical operations i.e. >, <, >=, <=, ==, !=, `&` and `|`.
    The only difference is that not only the BandMath expression is saved (e.g. "im1b1 > 0 ? 1 : 0"), but also the
    logical expression (e.g. "im1b1 > 0")
    """

    def __init__(self, operator, *inputs, nb_bands=None):
        super().__init__(operator, *inputs, nb_bands=nb_bands)

        self.logical_exp_bands, self.logical_exp = self.get_real_exp(self.logical_fake_exp_bands)

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        self.inputs = []
        self.nb_channels = {}

        # For any other operations, the output number of bands is the same as inputs
        if any([isinstance(input, Slicer) and hasattr(input, 'one_band_sliced') for input in inputs]):
            nb_bands = 1
        else:
            nb_bands_list = [get_nbchannels(input) for input in inputs if not isinstance(input, (float, int))]
            # check that all inputs have the same nb of bands
            if len(nb_bands_list) > 1:
                if not all(x == nb_bands_list[0] for x in nb_bands_list):
                    raise Exception('All images do not have the same number of bands')
            nb_bands = nb_bands_list[0]

        # Create a list of fake exp, each item of the list corresponding to one band
        self.fake_exp_bands = []
        self.logical_fake_exp_bands = []
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for input in inputs:
                fake_exp, corresponding_inputs, nb_channels = self._create_one_input_fake_exp(input, band,
                                                                                              keep_logical=True)

                fake_exps.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresponding_inputs and nb_channels:
                    self.inputs.extend(corresponding_inputs)
                    self.nb_channels.update(nb_channels)

            # We create here the "fake" expression. For example, for a BandMathX expression such as 'im1 > im2',
            # the logical fake expression stores the expression "str(input1) > str(input2)"
            logical_fake_exp = f'({fake_exps[0]} {operator} {fake_exps[1]})'

            # We keep the logical expression, useful if later combined with other logical operations
            self.logical_fake_exp_bands.append(logical_fake_exp)
            # We create a valid BandMath expression, e.g. "str(input1) > str(input2) ? 1 : 0"
            fake_exp = f'({logical_fake_exp} ? 1 : 0)'
            self.fake_exp_bands.append(fake_exp)


def get_nbchannels(inp):
    """
    Get the nb of bands of input image
    :param inp: a str
    :return: number of bands in image
    """
    if isinstance(inp, otbObject):
        nb_channels = inp.shape[-1]
    else:
        # Executing the app, without printing its log
        try:
            info = App("ReadImageInfo", inp, otb_stdout=False)
            nb_channels = info.GetParameterInt("numberbands")
        except (RuntimeError, TypeError) as e:
            logger.error(f'Not a valid image : {inp}')
            logger.error(f'{e}')
            nb_channels = None
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
        info = App("ReadImageInfo", inp, otb_stdout=False)
        datatype = info.GetParameterInt("datatype")  # which is such as short, float...
        dataype_to_pixeltype = {
            'unsigned_char': 'uint8', 'short': 'int16', 'unsigned_short': 'uint16', 
            'int': 'int32', 'unsigned_int': 'uint32', 'long': 'int32', 'ulong': 'uint32', 
            'float': 'float','double': 'double'
        }
        pixel_type = dataype_to_pixeltype[datatype]
        pixel_type = getattr(otb, f'ImagePixelType_{pixel_type}')
    elif isinstance(inp, (Input, Output, Operation)):
        pixel_type = inp.GetImageBasePixelType(inp.output_parameter_key)
    elif isinstance(inp, App):
        if len(inp.output_parameters_keys) > 1:
            pixel_type = inp.GetImageBasePixelType(inp.output_parameters_keys[0])
        else:
            pixel_type = {key: inp.GetImageBasePixelType(key) for key in inp.output_parameters_keys}

    return pixel_type
