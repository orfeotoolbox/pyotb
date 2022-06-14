# -*- coding: utf-8 -*-
"""
This module is the core of pyotb
"""
from abc import ABC
from pathlib import Path

import numpy as np
import otbApplication as otb

from .helpers import logger


class otbObject(ABC):
    """Abstract class that gathers common operations for any OTB in-memory raster.
    All child of this class must have an `app` attribute that is an OTB application.
    """

    @property
    def output_param(self):
        """
        Property to get object's unique (or first) output parameter key

        Returns:
           output parameter key

        """
        if hasattr(self, 'output_parameter_key'):  # this is for Input, Output, Operation, Slicer
            return self.output_parameter_key
        # this is for App
        if not self.output_parameters_keys:
            return ""  # apps without outputs
        return self.output_parameters_keys[0]

    @property
    def shape(self):
        """
        Enables to retrieve the shape of a pyotb object.

        Returns:
            shape: (width, height, bands)

        """
        self.execute_if_needed()
        image_size = self.app.GetImageSize(self.output_param)
        image_bands = self.app.GetImageNbBands(self.output_param)
        return (*image_size, image_bands)

    def write(self, *args, filename_extension="", pixel_type=None, **kwargs):
        """
        Trigger execution (always), set output pixel type and write the output

        Args:
            *args: Can be : - dictionary containing key-arguments enumeration. Useful when a key contains
                              non-standard characters such as a point, e.g. {'io.out':'output.tif'}
                            - string, useful when there is only one output, e.g. 'output.tif'
                            - None if output file was passed during App init
            filename_extension: Optional, an extended filename as understood by OTB (e.g. "&gdal:co:TILED=YES")
                                Will be used for all outputs (Default value = "")
            pixel_type: Can be : - dictionary {output_parameter_key: pixeltype} when specifying for several outputs
                                 - str (e.g. 'uint16') or otbApplication.ImagePixelType_... When there are several
                                   outputs, all outputs are written with this unique type.
                                   Valid pixel types are uint8, uint16, uint32, int16, int32, float, double,
                                   cint16, cint32, cfloat, cdouble. (Default value = None)
            **kwargs: keyword arguments e.g. out='output.tif'

        Returns:
            boolean flag that indicate if command executed with success

        """
        # Gather all input arguments in kwargs dict
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, str) and kwargs:
                logger.warning('%s: Keyword arguments specified, ignoring argument "%s"', self.name, arg)
            elif isinstance(arg, str):
                kwargs.update({self.output_param: arg})

        dtypes = {}
        if isinstance(pixel_type, dict):
            dtypes = {k: parse_pixel_type(v) for k, v in pixel_type.items()}
        elif pixel_type is not None:
            typ = parse_pixel_type(pixel_type)
            if isinstance(self, App):
                dtypes = {key: typ for key in self.output_parameters_keys}
            elif hasattr(self, 'output_parameter_key'):
                dtypes = {self.output_parameter_key: typ}

        # Case output parameter was set during App init
        if not kwargs:
            if isinstance(self, App):
                if self.output_param in self.parameters:
                    if dtypes:
                        self.app.SetParameterOutputImagePixelType(self.output_param, dtypes[self.output_param])
                    return self.execute()
            raise ValueError(f'{self.app.GetName()}: Output parameter is missing.')

        if filename_extension:
            logger.debug('%s: Using extended filename for outputs: %s', self.name, filename_extension)
            if not filename_extension.startswith('?'):
                filename_extension = "?" + filename_extension

        # Parse kwargs
        for key, output_filename in kwargs.items():
            # Stop process if a bad parameter is given
            if not self.__check_output_param(key):
                raise KeyError(f'{self.app.GetName()}: Wrong parameter key "{key}"')
            # Check if extended filename was not provided twice
            if '?' in output_filename and filename_extension:
                logger.warning('%s: Extended filename was provided twice. Using the one found in path.', self.name)
            elif filename_extension:
                output_filename += filename_extension

            logger.debug('%s: "%s" parameter is %s', self.name, key, output_filename)
            if isinstance(self, App):
                self.set_parameters({key: output_filename})
            else:
                self.app.SetParameterString(key, output_filename)
            if key in dtypes:
                self.app.SetParameterOutputImagePixelType(key, dtypes[key])

        self.app.PropagateConnectMode(False)

        if isinstance(self, App):
            return self.execute()

        self.app.Execute()
        self.app.WriteOutput()
        return True

    def to_numpy(self, propagate_pixel_type=True, copy=False):
        """
        Export a pyotb object to numpy array

        Args:
            propagate_pixel_type: when set to True, the numpy array is created with the same pixel type as
                                  the otbObject first output. Default is True.
            copy: whether to copy the output array, default is False
                  required to True if propagate_pixel_type is False and the source app reference is lost

        Returns:
          a numpy array

        """
        self.execute_if_needed()
        array = self.app.ExportImage(self.output_param)['array']
        if propagate_pixel_type:
            otb_pixeltype = get_pixel_type(self)
            otb_pixeltype_to_np_pixeltype = {0: np.uint8, 1: np.int16, 2: np.uint16, 3: np.int32,
                                             4: np.uint32, 5: np.float32, 6: np.float64}
            np_pixeltype = otb_pixeltype_to_np_pixeltype[otb_pixeltype]
            return array.astype(np_pixeltype)
        if copy:
            return array.copy()
        return array

    def __check_output_param(self, key):
        """
        Check param name to prevent strange behaviour in write() if kwarg key is not implemented

        Args:
            key: parameter key

        Returns:
            bool flag
        """
        if hasattr(self, 'output_parameter_key'):
            return key == self.output_parameter_key
        return key in self.output_parameters_keys

    def execute_if_needed(self):
        """
        Execute the app if it hasn't been already executed
        """
        if isinstance(self, App):
            if not self.finished:
                self.execute()
        elif isinstance(self, Output):
            self.app.Execute()

    # Special methods
    def __getitem__(self, key):
        """
        This function enables 2 things :
        - access attributes like that : object['any_attribute']
        - slicing, i.e. selecting ROI/bands. For example, selecting first 3 bands: object[:, :, :3]
                                                          selecting bands 1, 2 & 5 : object[:, :, [0, 1, 4]]
                                                          selecting 1000x1000 subset : object[:1000, :1000]

        Args:
            key: attribute key

        Returns:
            attribute or Slicer
        """
        # Accessing string attributes
        if isinstance(key, str):
            return self.__dict__.get(key)

        # Slicing
        if not isinstance(key, tuple) or (isinstance(key, tuple) and (len(key) < 2 or len(key) > 3)):
            raise ValueError(f'"{key}"cannot be interpreted as valid slicing. Slicing should be 2D or 3D.')
        if isinstance(key, tuple) and len(key) == 2:
            # Adding a 3rd dimension
            key = key + (slice(None, None, None),)
        (cols, rows, channels) = key
        return Slicer(self, rows, cols, channels)

    def __getattr__(self, name):
        """
        This method is called when the default attribute access fails. We choose to access the attribute `name`
        of self.app. Thus, any method of otbApplication can be used transparently on otbObject objects,
        e.g. SetParameterOutputImagePixelType() or ExportImage() work

        Args:
            name: attribute name

        Returns:
            attribute

        Raises:
            AttributeError: when `name` is not an attribute of self.app

        """
        try:
            res = getattr(self.app, name)
            return res
        except AttributeError as e:
            raise AttributeError(f'{self.name}: Could not find attribute `{name}`') from e

    def __add__(self, other):
        """
        Overrides the default addition and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self + other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', self, other)

    def __sub__(self, other):
        """
        Overrides the default subtraction and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self - other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', self, other)

    def __mul__(self, other):
        """
        Overrides the default subtraction and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self * other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', self, other)

    def __truediv__(self, other):
        """
        Overrides the default subtraction and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self / other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', self, other)

    def __radd__(self, other):
        """
        Overrides the default reverse addition and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             other + self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', other, self)

    def __rsub__(self, other):
        """
        Overrides the default subtraction and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             other - self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', other, self)

    def __rmul__(self, other):
        """
        Overrides the default multiplication and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             other * self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', other, self)

    def __rtruediv__(self, other):
        """
        Overrides the default division and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             other / self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', other, self)

    def __abs__(self):
        """
        Overrides the default abs operator and flavours it with BandMathX

        Returns:
            abs(self)

        """
        return Operation('abs', self)

    def __ge__(self, other):
        """
        Overrides the default greater or equal and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self >= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>=', self, other)

    def __le__(self, other):
        """
        Overrides the default less or equal and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self <= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<=', self, other)

    def __gt__(self, other):
        """
        Overrides the default greater operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self > other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>', self, other)

    def __lt__(self, other):
        """
        Overrides the default less operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self < other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<', self, other)

    def __eq__(self, other):
        """
        Overrides the default eq operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self == other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('==', self, other)

    def __ne__(self, other):
        """
        Overrides the default different operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self != other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('!=', self, other)

    def __or__(self, other):
        """
        Overrides the default or operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self || other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('||', self, other)

    def __and__(self, other):
        """
        Overrides the default and operator and flavours it with BandMathX

        Args:
            other: the other member of the operation

        Returns:
             self && other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('&&', self, other)

    # TODO: other operations ?
    #  e.g. __pow__... cf https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __hash__(self):
        """
        Returns:
            self hash

        """
        return id(self)

    def __array__(self):
        """
        This is called when running np.asarray(pyotb_object)

        Returns
            a numpy array

        """
        return self.to_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        This is called whenever a numpy function is called on a pyotb object. Operation is performed in numpy, then
        imported back to pyotb with the same georeference as input.

        Args:
            ufunc: numpy function
            method: an internal numpy argument
            inputs: inputs, at least one being pyotb object. If there are several pyotb objects, they must all have
                    the same georeference and pixel size.
            **kwargs: kwargs of the numpy function

        Returns:
            a pyotb object

        """
        if method == '__call__':
            # Converting potential pyotb inputs to arrays
            arrays = []
            image_dic = None
            for inp in inputs:
                if isinstance(inp, (float, int, np.ndarray, np.generic)):
                    arrays.append(inp)
                elif isinstance(inp, otbObject):
                    inp.execute_if_needed()
                    image_dic = inp.app.ExportImage(self.output_param)
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
            app = App('ExtractROI', image_dic=result_dic)  # pass the result_dic just to keep reference
            if result_array.shape[2] == 1:
                app.ImportImage('in', result_dic)
            else:
                app.ImportVectorImage('in', result_dic)
            app.execute()
            return app

        return NotImplemented


class Slicer(otbObject):
    """Slicer objects i.e. when we call something like raster[:, :, 2] from Python"""

    def __init__(self, x, rows, cols, channels):
        """
        Create a slicer object, that can be used directly for writing or inside a BandMath :
        - an ExtractROI app that handles extracting bands and ROI and can be written to disk or used in pipelines
        - in case the user only wants to extract one band, an expression such as "im1b#"

        Args:
            x: input
            rows: rows slicing (e.g. 100:2000)
            cols: columns slicing (e.g. 100:2000)
            channels: channels, can be slicing, list or int

        """
        # Initialize the app that will be used for writing the slicer
        self.output_parameter_key = 'out'
        self.name = 'Slicer'
        # Trigger source app execution if needed
        x.execute_if_needed()
        app = App('ExtractROI', {'in': x, 'mode': 'extent'}, propagate_pixel_type=True)
        # First exec required in order to read image dim
        app.app.Execute()

        parameters = {}
        # Channel slicing
        nb_channels = get_nbchannels(x)
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
            parameters.update({'cl': [f'Channel{i + 1}' for i in channels]})

        # Spatial slicing
        spatial_slicing = False
        # TODO: handle PixelValue app so that accessing value is possible, e.g. raster[120, 200, 0]
        # TODO TBD: handle the step value in the slice so that NN undersampling is possible ? e.g. obj[::2, ::2]
        if rows.start is not None:
            parameters.update({'mode.extent.uly': rows.start})
            spatial_slicing = True
        if rows.stop is not None and rows.stop != -1:
            parameters.update(
                {'mode.extent.lry': rows.stop - 1})  # subtract 1 to be compliant with python convention
            spatial_slicing = True
        if cols.start is not None:
            parameters.update({'mode.extent.ulx': cols.start})
            spatial_slicing = True
        if cols.stop is not None and cols.stop != -1:
            parameters.update(
                {'mode.extent.lrx': cols.stop - 1})  # subtract 1 to be compliant with python convention
            spatial_slicing = True
        # Execute app
        app.set_parameters(**parameters)
        app.execute()
        # Keeping the OTB app, not the pyotb app
        self.app = app.app

        # These are some attributes when the user simply wants to extract *one* band to be used in an Operation
        if not spatial_slicing and isinstance(channels, list) and len(channels) == 1:
            self.one_band_sliced = channels[0] + 1  # OTB convention: channels start at 1
            self.input = x


class Input(otbObject):
    """
    Class for transforming a filepath to pyOTB object
    """

    def __init__(self, filepath):
        """
        Args:
            filepath: raster file path

        """
        self.output_parameter_key = 'out'
        self.filepath = filepath
        self.name = f'Input from {filepath}'
        app = App('ExtractROI', filepath, execute=True, propagate_pixel_type=True)
        self.app = app.app

    def __str__(self):
        """
        Returns:
            string representation

        """
        return f'<pyotb.Input object from {self.filepath}>'


class Output(otbObject):
    """
    Class for output of an app
    """

    def __init__(self, app, output_parameter_key):
        """
        Args:
            app: The OTB application
            output_parameter_key: Output parameter key

        """
        self.app = app  # keeping a reference of the OTB app
        self.output_parameter_key = output_parameter_key
        self.name = f'Output {output_parameter_key} from {self.app.GetName()}'

    def __str__(self):
        """
        Returns:
            string representation

        """
        return f'<pyotb.Output {self.app.GetName()} object, id {id(self)}>'


class App(otbObject):
    """
    Class of an OTB app
    """
    _name = ""

    @property
    def name(self):
        """
        Returns:
            user's defined name or appname

        """
        return self._name or self.appname

    @name.setter
    def name(self, val):
        """Set custom App name

        Args:
          val: new name

        """
        self._name = val

    @property
    def finished(self):
        """
        Property to store whether App has been executed but False if any output file is missing

        Returns:
            True if exec ended and output files are found else False

        """
        if self._ended and self.find_output():
            return True
        return False

    @finished.setter
    def finished(self, val):
        """
        Value `_ended` will be set to True right after App.execute() or App.write(),
        then find_output() is called when accessing the property

        Args:
            val: True if execution ended without exceptions

        """
        self._ended = val

    def __init__(self, appname, *args, execute=False, image_dic=None, otb_stdout=True,
                 pixel_type=None, propagate_pixel_type=False, **kwargs):
        """
        Enables to init an OTB application as a oneliner. Handles in-memory connection between apps.

        Args:
            appname: name of the app, e.g. 'Smoothing'
            *args: Used for passing application parameters. Can be :
                           - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                             (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                           - string, App or Output, useful when the user wants to specify the input "in"
                           - list, useful when the user wants to specify the input list 'il'
            execute: whether to Execute the app. False is default and required when not all mandatory params are set
                     Should be set to True when creating an app as a oneliner.
            image_dic: optional. Enables to keep a reference to image_dic. image_dic is a dictionary, such as
                       the result of app.ExportImage(). Use it when the app takes a numpy array as input.
                       See this related issue for why it is necessary to keep reference of object:
                       https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/1824
            otb_stdout: whether to print logs of the OTB app
            pixel_type: Can be : - dictionary {output_parameter_key: pixeltype} when specifying for several outputs
                                 - str (e.g. 'uint16') or otbApplication.ImagePixelType_... When there are several
                                   outputs, all outputs are written with this unique type
                                   Valid pixel types are double, float, uint8, uint16, uint32, int16, int32,
                                   cint16, cint32, cfloat, cdouble.
            propagate_pixel_type: propagate the pixel type from inputs to output. If several inputs, the type of an
                                  arbitrary input is considered. If several outputs, all will have the same type.
            **kwargs: Used for passing application parameters.
                      e.g. il=['input1.tif', App_object2, App_object3.out], out='output.tif'

        """
        self.appname = appname
        if otb_stdout:
            self.app = otb.Registry.CreateApplication(appname)
        else:
            self.app = otb.Registry.CreateApplicationWithoutLogger(appname)
        self.image_dic = image_dic
        self.preserve_dtype = propagate_pixel_type
        self._ended = False
        # Parameters
        self.parameters = {}
        self.output_parameters_keys = self.get_output_parameters_keys()
        if not (args or kwargs):
            return
        self.set_parameters(*args, **kwargs)
        # Output pixel type
        dtypes = {}
        if isinstance(pixel_type, dict):
            dtypes = {k: parse_pixel_type(v) for k, v in pixel_type.items()}
        elif pixel_type is not None:
            dtypes = {key: parse_pixel_type(pixel_type) for key in self.output_parameters_keys}
        for key, typ in dtypes.items():
            self.app.SetParameterOutputImagePixelType(key, typ)
        # Here we make sure that intermediate outputs will be flushed to disk
        if self.__with_output():
            self.app.PropagateConnectMode(False)
        # Run app, write output if needed, update `finished` property
        if execute or not self.output_param:
            self.execute()
        # Force save param because it wasn't called during execute()
        else:
            self.__save_objects()

    def execute(self):
        """
        Execute with appropriate and outputs to disk if any output parameter was set

        Returns:
             boolean flag that indicate if command executed with success

        """
        success = False
        logger.debug("%s: run execute() with parameters=%s", self.name, self.parameters)
        try:
            self.app.Execute()
            if self.__with_output():
                self.app.WriteOutput()
            self.finished = True
        except (RuntimeError, FileNotFoundError) as e:
            raise Exception(f'{self.name}: error during during app execution') from e
        self.__save_objects()
        success = self.finished
        if success:
            logger.debug("%s: execution succeeded", self.name)

        return success

    def find_output(self):
        """
        Find output files on disk using parameters

        Returns:
            list of files found on disk

        """
        if not self.__with_output():
            return [""]
        # Return non-empty list to make sure finished returns True
        files = []
        missing = []
        outputs = [p for p in self.output_parameters_keys if p in self.parameters]
        for param in outputs:
            filename = self.parameters[param]
            # Remove filename extension
            if '?' in filename:
                filename = filename.split('?')[0]
            path = Path(filename)
            if path.exists():
                files.append(str(path.absolute()))
            else:
                missing.append(str(path.absolute()))
        if missing:
            missing = tuple(missing)
            for filename in missing:
                logger.error("%s: execution seems to have failed, %s does not exist", self.name, filename)
        else:
            # Force set _ended to True since Execute was triggered by OTB, execute() was not called
            self.finished = True

        return files

    def clear(self, parameters=False, memory=False, key=""):
        """Free resources and reset App state

        Args:
            parameters: to clear settings dictionary (Default value = False)
            memory: to free app resources in memory (Default value = False)
            key: to clear one specific parameter value (Default value = "")

        """
        if parameters:
            for p in self.parameters:
                self.app.ClearValue(p)
        elif key:
            self.app.ClearValue(key)
        if memory:
            self.app.FreeRessources()

    def get_output_parameters_keys(self):
        """Get raster output parameter keys

        Returns:
            output parameters keys
        """
        return [param for param in self.app.GetParametersKeys()
                if self.app.GetParameterType(param) == otb.ParameterType_OutputImage]

    def set_parameters(self, *args, **kwargs):
        """Set some parameters of the app. When useful, e.g. for images list, this function appends the parameters
        instead of overwriting them. Handles any parameters, i.e. in-memory & filepaths

        Args:
            *args: Can be : - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                              (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                            - string, App or Output, useful when the user implicitly wants to set the param "in"
                            - list, useful when the user implicitly wants to set the param "il"
            **kwargs: keyword arguments e.g. il=['input1.tif', oApp_object2, App_object3.out], out='output.tif'

        Raises:
            Exception: when the setting of a parameter failed

        """
        parameters = kwargs
        parameters.update(self.__parse_args(args))
        # Going through all arguments
        for param, obj in parameters.items():
            if param not in self.app.GetParametersKeys():
                raise Exception(f"{self.name}: parameter '{param}' was not recognized. "
                                f"Available keys are {self.app.GetParametersKeys()}")
            # When the parameter expects a list, if needed, change the value to list
            if self.__is_key_list(param) and not isinstance(obj, (list, tuple)):
                parameters[param] = [obj]
                obj = [obj]
                logger.warning('%s: Argument for parameter "%s" was converted to list', self.name, param)
            try:
                # This is when we actually call self.app.SetParameter*
                self.__set_param(param, obj)
            except (RuntimeError, TypeError, ValueError, KeyError) as e:
                raise Exception(f"{self.name}: something went wrong before execution "
                                f"(while setting parameter {param} to '{obj}')") from e

        # Update App's parameters attribute
        self.parameters.update(parameters)
        if self.preserve_dtype:
            self.__propagate_pixel_type()

    # Private functions
    @staticmethod
    def __parse_args(args):
        """
        Gather all input arguments in kwargs dict
        """
        kwargs = {}
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, (str, otbObject)):
                kwargs.update({'in': arg})
            elif isinstance(arg, list):
                kwargs.update({'il': arg})
        return kwargs

    def __set_param(self, param, obj):
        """
        Set one parameter, decide which otb.Application method to use depending on target object
        """
        # Single-parameter cases
        if isinstance(obj, otbObject):
            self.app.ConnectImage(param, obj.app, obj.output_param)
        elif isinstance(obj, otb.Application):  # this is for backward comp with plain OTB
            outparamkey = [param for param in obj.GetParametersKeys()
                           if obj.GetParameterType(param) == otb.ParameterType_OutputImage][0]
            self.app.ConnectImage(param, obj, outparamkey)
        elif param == 'ram':  # SetParameterValue in OTB<7.4 doesn't work for ram parameter cf gitlab OTB issue 2200
            self.app.SetParameterInt('ram', int(obj))
        elif not isinstance(obj, list):  # any other parameters (str, int...)
            self.app.SetParameterValue(param, obj)
        # Images list
        elif self.__is_key_images_list(param):
            # To enable possible in-memory connections, we go through the list and set the parameters one by one
            for inp in obj:
                if isinstance(inp, otbObject):
                    self.app.ConnectImage(param, inp.app, inp.output_param)
                elif isinstance(inp, otb.Application):  # this is for backward comp with plain OTB
                    outparamkey = [param for param in inp.GetParametersKeys() if
                                   inp.GetParameterType(param) == otb.ParameterType_OutputImage][0]
                    self.app.ConnectImage(param, inp, outparamkey)
                else:  # here `input` should be an image filepath
                    # Append `input` to the list, do not overwrite any previously set element of the image list
                    self.app.AddParameterStringList(param, inp)
        # List of any other types (str, int...)
        else:
            self.app.SetParameterValue(param, obj)

    def __propagate_pixel_type(self):
        """
        Propagate the pixel type from inputs to output. If several inputs, the type of an arbitrary input
        is considered. If several outputs, all outputs will have the same type.
        """
        pixel_type = None
        for param in self.parameters.values():
            try:
                pixel_type = get_pixel_type(param)
            except TypeError:
                pass
        if pixel_type is None:
            logger.warning("%s: Could not propagate pixel type from inputs to output, no valid input found", self.name)
        else:
            for out_key in self.output_parameters_keys:
                self.app.SetParameterOutputImagePixelType(out_key, pixel_type)

    def __with_output(self):
        """
        Check if App has any output parameter key
        """
        types = (otb.ParameterType_OutputFilename, otb.ParameterType_OutputImage, otb.ParameterType_OutputVectorData)
        params = [param for param in self.app.GetParametersKeys() if self.app.GetParameterType(param) in types]
        return any(key in self.parameters for key in params)

    def __save_objects(self):
        """
        Saving app parameters and outputs as attributes, so that they can be accessed with `obj.key`
        which is useful when the key contains reserved characters such as a point eg "io.out"
        """
        for key in self.app.GetParametersKeys():
            if key in self.output_parameters_keys:  # raster outputs
                output = Output(self.app, key)
                setattr(self, key, output)
            elif key not in ('parameters',):  # any other attributes (scalars...)
                try:
                    setattr(self, key, self.app.GetParameterValue(key))
                except RuntimeError:
                    pass  # this is when there is no value for key

    def __is_key_list(self, key):
        """
        Check if a key of the App is an input parameter list
        """
        return self.app.GetParameterType(key) in (
            otb.ParameterType_InputImageList,
            otb.ParameterType_StringList,
            otb.ParameterType_InputFilenameList,
            otb.ParameterType_InputVectorDataList,
            otb.ParameterType_ListView
        )

    def __is_key_images_list(self, key):
        """
        Check if a key of the App is an input parameter image list
        """
        return self.app.GetParameterType(key) in (
            otb.ParameterType_InputImageList,
            otb.ParameterType_InputFilenameList
        )

    # Special methods
    def __str__(self):
        """
        Return a nice str
        """
        return f'<pyotb.App {self.appname} object id {id(self)}>'


class Operation(otbObject):
    """
    Class for arithmetic/math operations done in Python.

    Example:
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

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            *inputs: inputs. Can be App, Output, Input, Operation, Slicer, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        self.operator = operator
        for inp in inputs:
            if isinstance(inp, otbObject):
                inp.execute_if_needed()
        # We first create a 'fake' expression. E.g for the operation `input1 + input2` , we create a fake expression
        # that is like "str(input1) + str(input2)"
        self.inputs = []
        self.nb_channels = {}
        self.fake_exp_bands = []
        self.logical_fake_exp_bands = []

        self.create_fake_exp(operator, inputs, nb_bands=nb_bands)

        # Transforming images to the adequate im#, e.g. `input1` to "im1"
        # creating a dictionary that is like {str(input1): 'im1', 'image2.tif': 'im2', ...}.
        # NB: the keys of the dictionary are strings-only, instead of 'complex' objects, to enable easy serialization
        self.im_dic = {}
        self.im_count = 1
        mapping_str_to_input = {}  # to be able to retrieve the real python object from its string representation
        for inp in self.inputs:
            if not isinstance(inp, (int, float)):
                if str(inp) not in self.im_dic:
                    self.im_dic[str(inp)] = f'im{self.im_count}'
                    mapping_str_to_input[str(inp)] = inp
                    self.im_count += 1

        # getting unique image inputs, in the order im1, im2, im3 ...
        self.unique_inputs = [mapping_str_to_input[str_input] for str_input in sorted(self.im_dic, key=self.im_dic.get)]
        self.output_parameter_key = 'out'

        # Computing the BandMath or BandMathX app
        self.exp_bands, self.exp = self.get_real_exp(self.fake_exp_bands)
        self.name = f'Operation exp="{self.exp}"'

        if len(self.exp_bands) == 1:
            app = App('BandMath', il=self.unique_inputs, exp=self.exp)
        else:
            app = App('BandMathX', il=self.unique_inputs, exp=self.exp)
        app.execute()
        self.app = app.app

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        """
        Create a 'fake' expression. E.g for the operation input1 + input2 , we create a fake expression
        that is like "str(input1) + str(input2)"

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            inputs: inputs. Can be App, Output, Input, Operation, Slicer, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        self.inputs.clear()
        self.nb_channels.clear()

        logger.debug("%s, %s", operator, inputs)
        # this is when we use the ternary operator with `pyotb.where` function. The output nb of bands is already known
        if operator == '?' and nb_bands:
            pass
        # For any other operations, the output number of bands is the same as inputs
        else:
            if any(isinstance(input, Slicer) and hasattr(input, 'one_band_sliced') for input in inputs):
                nb_bands = 1
            else:
                nb_bands_list = [get_nbchannels(input) for input in inputs if not isinstance(input, (float, int))]
                # check that all inputs have the same nb of bands
                if len(nb_bands_list) > 1:
                    if not all(x == nb_bands_list[0] for x in nb_bands_list):
                        raise Exception('All images do not have the same number of bands')
                nb_bands = nb_bands_list[0]

        # Create a list of fake expressions, each item of the list corresponding to one band
        self.fake_exp_bands.clear()
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for k, inp in enumerate(inputs):
                # Generating the fake expression of the current input
                # this is a special case for the condition of the ternary operator `cond ? x : y`
                if len(inputs) == 3 and k == 0:
                    # when cond is monoband whereas the result is multiband, we expand the cond to multiband
                    if nb_bands != inp.shape[2]:
                        cond_band = 1
                    else:
                        cond_band = band
                    fake_exp, corresponding_inputs, nb_channels = self.create_one_input_fake_exp(inp, cond_band,
                                                                                                 keep_logical=True)
                # any other input
                else:
                    fake_exp, corresponding_inputs, nb_channels = self.create_one_input_fake_exp(inp, band,
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

    def get_real_exp(self, fake_exp_bands):
        """
        Generates the BandMathX expression

        Args:
            fake_exp_bands: list of fake expressions, each item corresponding to one band

        Returns:
            exp_bands: BandMath expression, split in a list, each item corresponding to one band
            exp: BandMath expression

        """
        # Create a list of expression, each item corresponding to one band (e.g. ['im1b1 + 1', 'im1b2 + 1'])
        exp_bands = []
        for one_band_fake_exp in fake_exp_bands:
            one_band_exp = one_band_fake_exp
            for inp in self.inputs:
                # replace the name of in-memory object (e.g. '<pyotb.App object>b1' by 'im1b1')
                one_band_exp = one_band_exp.replace(str(inp), self.im_dic[str(inp)])
            exp_bands.append(one_band_exp)

        # Form the final expression (e.g. 'im1b1 + 1; im1b2 + 1')
        exp = ';'.join(exp_bands)

        return exp_bands, exp

    @staticmethod
    def create_one_input_fake_exp(x, band, keep_logical=False):
        """
        This an internal function, only to be used by `create_fake_exp`. Enable to create a fake expression just for one
        input and one band.

        Args:
            x: input
            band: which band to consider (bands start at 1)
            keep_logical: whether to keep the logical expressions "as is" in case the input is a logical operation.
                          ex: if True, for `input1 > input2`, returned fake expression is "str(input1) > str(input2)"
                          if False, for `input1 > input2`, returned fake exp is "str(input1) > str(input2) ? 1 : 0".
                          Default False

        Returns:
            fake_exp: the fake expression for this band and input
            inputs: if the input is an Operation, we returns its own inputs
            nb_channels: if the input is an Operation, we returns its own nb_channels

        """
        # Special case for one-band slicer
        if isinstance(x, Slicer) and hasattr(x, 'one_band_sliced'):
            if keep_logical and isinstance(x.input, logicalOperation):
                fake_exp = x.input.logical_fake_exp_bands[x.one_band_sliced - 1]
                inputs = x.input.inputs
                nb_channels = x.input.nb_channels
            elif isinstance(x.input, Operation):
                # keep only one band of the expression
                fake_exp = x.input.fake_exp_bands[x.one_band_sliced - 1]
                inputs = x.input.inputs
                nb_channels = x.input.nb_channels
            else:
                # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
                fake_exp = str(x.input) + f'b{x.one_band_sliced}'
                inputs = [x.input]
                nb_channels = {x.input: 1}
        # For logicalOperation, we save almost the same attributes as an Operation
        elif keep_logical and isinstance(x, logicalOperation):
            fake_exp = x.logical_fake_exp_bands[band - 1]
            inputs = x.inputs
            nb_channels = x.nb_channels
        elif isinstance(x, Operation):
            fake_exp = x.fake_exp_bands[band - 1]
            inputs = x.inputs
            nb_channels = x.nb_channels
        # For int or float input, we just need to save their value
        elif isinstance(x, (int, float)):
            fake_exp = str(x)
            inputs = None
            nb_channels = None
        # We go on with other inputs, i.e. pyotb objects, filepaths...
        else:
            nb_channels = {x: get_nbchannels(x)}
            inputs = [x]
            # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
            fake_exp = str(x) + f'b{band}'

        return fake_exp, inputs, nb_channels

    def __str__(self):
        """
        Returns:
            a nice string representation

        """
        return f'<pyotb.Operation `{self.operator}` object, id {id(self)}>'


class logicalOperation(Operation):
    """
    This is a specialization of Operation class for boolean logical operations i.e. >, <, >=, <=, ==, !=, `&` and `|`.
    The only difference is that not only the BandMath expression is saved (e.g. "im1b1 > 0 ? 1 : 0"), but also the
    logical expression (e.g. "im1b1 > 0")

    """

    def __init__(self, operator, *inputs, nb_bands=None):
        """
        Args:
            operator: string operator (one of >, <, >=, <=, ==, !=, &, |)
            *inputs: inputs
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        super().__init__(operator, *inputs, nb_bands=nb_bands)

        self.logical_exp_bands, self.logical_exp = self.get_real_exp(self.logical_fake_exp_bands)

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        """
        Create a 'fake' expression. E.g for the operation input1 > input2 , we create a fake expression
        that is like "str(input1) > str(input2) ? 1 : 0" and a logical fake expression that is like
        "str(input1) > str(input2)"

        Args:
            operator: str (one of >, <, >=, <=, ==, !=, &, |)
            inputs: Can be App, Output, Input, Operation, Slicer, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """

        # For any other operations, the output number of bands is the same as inputs
        if any(isinstance(input, Slicer) and hasattr(input, 'one_band_sliced') for input in inputs):
            nb_bands = 1
        else:
            nb_bands_list = [get_nbchannels(input) for input in inputs if not isinstance(input, (float, int))]
            # check that all inputs have the same nb of bands
            if len(nb_bands_list) > 1:
                if not all(x == nb_bands_list[0] for x in nb_bands_list):
                    raise Exception('All images do not have the same number of bands')
            nb_bands = nb_bands_list[0]

        # Create a list of fake exp, each item of the list corresponding to one band
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for inp in inputs:
                fake_exp, corresponding_inputs, nb_channels = super().create_one_input_fake_exp(inp, band,
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
    """Get the nb of bands of input image

    Args:
        inp: can be filepath or pyotb object

    Returns:
        number of bands in image

    """
    if isinstance(inp, otbObject):
        nb_channels = inp.shape[-1]
    else:
        # Executing the app, without printing its log
        try:
            info = App("ReadImageInfo", inp, execute=True, otb_stdout=False)
            nb_channels = info.GetParameterInt("numberbands")
        except Exception as e:  # this happens when we pass a str that is not a filepath
            raise TypeError(f'Could not get the number of channels of `{inp}`. Not a filepath or wrong filepath') from e
    return nb_channels


def get_pixel_type(inp):
    """Get the encoding of input image pixels

    Args:
        inp: can be filepath or pyotb object

    Returns
        pixel_type: format is like `otbApplication.ImagePixelType_uint8', which actually is an int. For an App
                    with several outputs, only the pixel type of the first output is returned

    """
    if isinstance(inp, str):
        # Executing the app, without printing its log
        try:
            info = App("ReadImageInfo", inp, execute=True, otb_stdout=False)
        except Exception as info_err:  # this happens when we pass a str that is not a filepath
            raise TypeError(f'Could not get the pixel type of `{inp}`. Not a filepath or wrong filepath') from info_err
        datatype = info.GetParameterString("datatype")  # which is such as short, float...
        if not datatype:
            raise Exception(f'Unable to read pixel type of image {inp}')
        datatype_to_pixeltype = {'unsigned_char': 'uint8', 'short': 'int16', 'unsigned_short': 'uint16',
                                 'int': 'int32', 'unsigned_int': 'uint32', 'long': 'int32', 'ulong': 'uint32',
                                 'float': 'float', 'double': 'double'}
        pixel_type = datatype_to_pixeltype[datatype]
        pixel_type = getattr(otb, f'ImagePixelType_{pixel_type}')
    elif isinstance(inp, (otbObject)):
        pixel_type = inp.GetParameterOutputImagePixelType(inp.output_param)
    else:
        raise TypeError(f'Could not get the pixel type. Not supported type: {inp}')

    return pixel_type


def parse_pixel_type(pixel_type):
    """Convert one str pixel type to OTB integer enum if necessary

    Args:
        pixel_type: pixel type. can be str, int or dict

    Returns:
        pixel_type integer value

    """
    if isinstance(pixel_type, str):  # this correspond to 'uint8' etc...
        return getattr(otb, f'ImagePixelType_{pixel_type}')
    if isinstance(pixel_type, int):
        return pixel_type
    raise ValueError(f'Bad pixel type specification ({pixel_type})')
