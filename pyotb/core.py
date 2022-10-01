# -*- coding: utf-8 -*-
"""This module is the core of pyotb."""
from pathlib import Path

import numpy as np
import otbApplication as otb

from .helpers import logger


class otbObject:
    """Base class that gathers common operations for any OTB in-memory raster."""
    _name = ""
    app = None
    output_param = ""

    @property
    def name(self):
        """Application name that will be printed in logs.

        Returns:
            user's defined name or appname

        """
        return self._name or self.app.GetName()

    @name.setter
    def name(self, val):
        """Set custom name.

        Args:
          val: new name

        """
        self._name = val

    @property
    def dtype(self):
        """Expose the pixel type of an output image using numpy convention.

        Returns:
            dtype: pixel type of the output image

        """
        try:
            enum = self.app.GetParameterOutputImagePixelType(self.output_param)
            return self.app.ConvertPixelTypeToNumpy(enum)
        except RuntimeError:
            return None

    @property
    def shape(self):
        """Enables to retrieve the shape of a pyotb object using numpy convention.

        Returns:
            shape: (height, width, bands)

        """
        width, height = self.app.GetImageSize(self.output_param)
        bands = self.app.GetImageNbBands(self.output_param)
        return (height, width, bands)

    def write(self, *args, filename_extension="", pixel_type=None, **kwargs):
        """Trigger execution, set output pixel type and write the output.

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
        """
        # Gather all input arguments in kwargs dict
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, str) and kwargs:
                logger.warning('%s: keyword arguments specified, ignoring argument "%s"', self.name, arg)
            elif isinstance(arg, str):
                kwargs.update({self.output_param: arg})

        dtypes = {}
        if isinstance(pixel_type, dict):
            dtypes = {k: parse_pixel_type(v) for k, v in pixel_type.items()}
        elif pixel_type is not None:
            typ = parse_pixel_type(pixel_type)
            if isinstance(self, App):
                dtypes = {key: typ for key in self.output_parameters_keys}
            elif isinstance(self, otbObject):
                dtypes = {self.output_param: typ}

        if filename_extension:
            logger.debug('%s: using extended filename for outputs: %s', self.name, filename_extension)
            if not filename_extension.startswith('?'):
                filename_extension = "?" + filename_extension

        # Case output parameter was set during App init
        if not kwargs:
            if self.output_param in self.parameters:
                if dtypes:
                    self.app.SetParameterOutputImagePixelType(self.output_param, dtypes[self.output_param])
                if filename_extension:
                    new_val = self.parameters[self.output_param] + filename_extension
                    self.app.SetParameterString(self.output_param, new_val)
            else:
                raise ValueError(f'{self.app.GetName()}: Output parameter is missing.')

        # Parse kwargs
        for key, output_filename in kwargs.items():
            # Stop process if a bad parameter is given
            if key not in self.app.GetParametersKeys():
                raise KeyError(f'{self.app.GetName()}: Unknown parameter key "{key}"')
            # Check if extended filename was not provided twice
            if '?' in output_filename and filename_extension:
                logger.warning('%s: extended filename was provided twice. Using the one found in path.', self.name)
            elif filename_extension:
                output_filename += filename_extension

            logger.debug('%s: "%s" parameter is %s', self.name, key, output_filename)
            self.app.SetParameterString(key, output_filename)

            if key in dtypes:
                self.app.SetParameterOutputImagePixelType(key, dtypes[key])

        logger.debug('%s: flushing data to disk', self.name)
        try:
            self.app.WriteOutput()
        except RuntimeError:
            logger.debug('%s: failed to simply write output, executing once again then writing', self.name)
            self.app.ExecuteAndWriteOutput()

    def to_numpy(self, preserve_dtype=True, copy=False):
        """Export a pyotb object to numpy array.

        Args:
            preserve_dtype: when set to True, the numpy array is created with the same pixel type as
                                  the otbObject first output. Default is True.
            copy: whether to copy the output array, default is False
                  required to True if preserve_dtype is False and the source app reference is lost

        Returns:
          a numpy array

        """
        array = self.app.ExportImage(self.output_param)['array']
        if preserve_dtype:
            return array.astype(self.dtype)
        if copy:
            return array.copy()
        return array

    def to_rasterio(self):
        """Export image as a numpy array and its metadata compatible with rasterio.

        Returns:
          array : a numpy array in the (bands, height, width) order
          profile: a metadata dict required to write image using rasterio

        """
        array = self.to_numpy(preserve_dtype=True, copy=False)
        array = np.moveaxis(array, 2, 0)
        proj = self.app.GetImageProjection(self.output_param)
        spacing_x, spacing_y = self.app.GetImageSpacing(self.output_param)
        origin_x, origin_y = self.app.GetImageOrigin(self.output_param)
        # Shift image origin since OTB is giving coordinates of pixel center instead of corners
        origin_x, origin_y = origin_x - spacing_x / 2, origin_y - spacing_y / 2
        profile = {
            'crs': proj, 'dtype': array.dtype,
            'count': array.shape[0], 'height': array.shape[1], 'width': array.shape[2],
            'transform': (spacing_x, 0.0, origin_x, 0.0, spacing_y, origin_y)  # here we force pixel rotation to 0 !
        }
        return array, profile

    # Special methods
    def __getitem__(self, key):
        """Override the default __getitem__ behaviour.

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
        (rows, cols, channels) = key
        return Slicer(self, rows, cols, channels)

    def __getattr__(self, name):
        """This method is called when the default attribute access fails.

        We choose to access the attribute `name` of self.app.
        Thus, any method of otbApplication can be used transparently on otbObject objects,
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
            raise AttributeError(f'{self.name}: could not find attribute `{name}`') from e

    def __add__(self, other):
        """Overrides the default addition and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self + other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', self, other)

    def __sub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self - other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', self, other)

    def __mul__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self * other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', self, other)

    def __truediv__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self / other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', self, other)

    def __radd__(self, other):
        """Overrides the default reverse addition and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             other + self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('+', other, self)

    def __rsub__(self, other):
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             other - self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('-', other, self)

    def __rmul__(self, other):
        """Overrides the default multiplication and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             other * self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('*', other, self)

    def __rtruediv__(self, other):
        """Overrides the default division and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             other / self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation('/', other, self)

    def __abs__(self):
        """Overrides the default abs operator and flavours it with BandMathX.

        Returns:
            abs(self)

        """
        return Operation('abs', self)

    def __ge__(self, other):
        """Overrides the default greater or equal and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self >= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>=', self, other)

    def __le__(self, other):
        """Overrides the default less or equal and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self <= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<=', self, other)

    def __gt__(self, other):
        """Overrides the default greater operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self > other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('>', self, other)

    def __lt__(self, other):
        """Overrides the default less operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self < other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('<', self, other)

    def __eq__(self, other):
        """Overrides the default eq operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self == other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('==', self, other)

    def __ne__(self, other):
        """Overrides the default different operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self != other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('!=', self, other)

    def __or__(self, other):
        """Overrides the default or operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
             self || other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return logicalOperation('||', self, other)

    def __and__(self, other):
        """Overrides the default and operator and flavours it with BandMathX.

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
        """Override the default behaviour of the hash function.

        Returns:
            self hash

        """
        return id(self)

    def __array__(self):
        """This is called when running np.asarray(pyotb_object).

        Returns:
            a numpy array

        """
        return self.to_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """This is called whenever a numpy function is called on a pyotb object.

        Operation is performed in numpy, then imported back to pyotb with the same georeference as input.

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
            app = App('ExtractROI', frozen=True, image_dic=result_dic)  # pass the result_dic just to keep reference
            if result_array.shape[2] == 1:
                app.ImportImage('in', result_dic)
            else:
                app.ImportVectorImage('in', result_dic)
            app.execute()
            return app

        return NotImplemented

    def summarize(self):
        """Return a nested dictionary summarizing the otbObject.

        Returns:
            Nested dictionary summarizing the otbObject

        """
        params = self.parameters
        for k, p in params.items():
            # In the following, we replace each parameter which is an otbObject, with its summary.
            if isinstance(p, otbObject):  # single parameter
                params[k] = p.summarize()
            elif isinstance(p, list):  # parameter list
                params[k] = [pi.summarize() if isinstance(pi, otbObject) else pi for pi in p]

        return {"name": self.name, "parameters": params}


class App(otbObject):
    """Class of an OTB app."""
    def __init__(self, appname, *args, frozen=False, quiet=False,
                 preserve_dtype=False, image_dic=None, **kwargs):
        """Enables to init an OTB application as a oneliner. Handles in-memory connection between apps.

        Args:
            appname: name of the app, e.g. 'Smoothing'
            *args: used for passing application parameters. Can be :
                           - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                             (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                           - string, App or Output, useful when the user wants to specify the input "in"
                           - list, useful when the user wants to specify the input list 'il'
            frozen: freeze OTB app in order to use execute() later and avoid blocking process during __init___
            quiet: whether to print logs of the OTB app
            preserve_dtype: propagate the pixel type from inputs to output. If several inputs, the type of an
                                  arbitrary input is considered. If several outputs, all will have the same type.
            image_dic: enables to keep a reference to image_dic. image_dic is a dictionary, such as
                       the result of app.ExportImage(). Use it when the app takes a numpy array as input.
                       See this related issue for why it is necessary to keep reference of object:
                       https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/1824
            **kwargs: used for passing application parameters.
                      e.g. il=['input1.tif', App_object2, App_object3.out], out='output.tif'

        """
        self.appname = appname
        self.frozen = frozen
        self.quiet = quiet
        self.preserve_dtype = preserve_dtype
        self.image_dic = image_dic
        if self.quiet:
            self.app = otb.Registry.CreateApplicationWithoutLogger(appname)
        else:
            self.app = otb.Registry.CreateApplication(appname)
        self.description = self.app.GetDocLongDescription()
        self.output_parameters_keys = self.__get_output_parameters_keys()
        if self.output_parameters_keys:
            self.output_param = self.output_parameters_keys[0]

        self.parameters = {}
        if (args or kwargs):
            self.set_parameters(*args, **kwargs)
        if not self.frozen:
            self.execute()

    def set_parameters(self, *args, **kwargs):
        """Set some parameters of the app.

        When useful, e.g. for images list, this function appends the parameters
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
                logger.warning('%s: argument for parameter "%s" was converted to list', self.name, param)
            try:
                # This is when we actually call self.app.SetParameter*
                self.__set_param(param, obj)
            except (RuntimeError, TypeError, ValueError, KeyError) as e:
                raise Exception(f"{self.name}: something went wrong before execution "
                                f"(while setting parameter '{param}' to '{obj}')") from e
        # Update _parameters using values from OtbApplication object
        otb_params = self.app.GetParameters().items()
        otb_params = {k: str(v) if isinstance(v, otb.ApplicationProxy) else v for k, v in otb_params}
        self.parameters.update({**parameters, **otb_params})
        # Update output images pixel types
        if self.preserve_dtype:
            self.__propagate_pixel_type()

    def execute(self):
        """Execute and write to disk if any output parameter has been set during init."""
        logger.debug("%s: run execute() with parameters=%s", self.name, self.parameters)
        try:
            self.app.Execute()
        except (RuntimeError, FileNotFoundError) as e:
            raise Exception(f'{self.name}: error during during app execution') from e
        self.frozen = False
        logger.debug("%s: execution ended", self.name)
        if self.__has_output_param_key():
            logger.debug('%s: flushing data to disk', self.name)
            self.app.WriteOutput()
        self.__save_objects()

    def find_output(self):
        """Find output files on disk using path found in parameters.

        Returns:
            list of files found on disk

        """
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

        return files

    # Private functions
    def __get_output_parameters_keys(self):
        """Get raster output parameter keys.

        Returns:
            output parameters keys
        """
        return [param for param in self.app.GetParametersKeys()
                if self.app.GetParameterType(param) == otb.ParameterType_OutputImage]

    def __has_output_param_key(self):
        """Check if App has any output parameter key."""
        if not self.output_param:
            return True  # apps like ReadImageInfo with no filetype output param still needs to WriteOutput
        types = (otb.ParameterType_OutputFilename, otb.ParameterType_OutputImage, otb.ParameterType_OutputVectorData)
        outfile_params = [param for param in self.app.GetParametersKeys() if self.app.GetParameterType(param) in types]
        return any(key in self.parameters for key in outfile_params)

    @staticmethod
    def __parse_args(args):
        """Gather all input arguments in kwargs dict.

        Returns:
            a dictionary with the right keyword depending on the object

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
        """Set one parameter, decide which otb.Application method to use depending on target object."""
        if obj is not None:
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
        """Propagate the pixel type from inputs to output.

        For several inputs, or with an image list, the type of the first input is considered.
        If several outputs, all outputs will have the same type.

        """
        pixel_type = None
        for key, param in self.parameters.items():
            if self.__is_key_input_image(key):
                if not param:
                    continue
                if isinstance(param, list):
                    param = param[0]  # first image in "il"
                try:
                    pixel_type = get_pixel_type(param)
                    type_name = self.app.ConvertPixelTypeToNumpy(pixel_type)
                    logger.debug('%s: output(s) will be written with type "%s"', self.name, type_name)
                    for out_key in self.output_parameters_keys:
                        self.app.SetParameterOutputImagePixelType(out_key, pixel_type)
                    return
                except TypeError:
                    pass

        logger.warning("%s: could not propagate pixel type from inputs to output, no valid input found", self.name)

    def __save_objects(self):
        """Saving app parameters and outputs as attributes, so that they can be accessed with `obj.key`.

        This is useful when the key contains reserved characters such as a point eg "io.out"
        """
        for key in self.app.GetParametersKeys():
            if key == 'parameters':  # skip forbidden attribute since it is already used by the App class
                continue
            value = None
            if key in self.output_parameters_keys:  # raster outputs
                value = Output(self, key)
            elif key in self.parameters:  # user or default app parameters
                value = self.parameters[key]
            else:  # any other app attribute (e.g. ReadImageInfo results)
                try:
                    value = self.app.GetParameterValue(key)
                except RuntimeError:
                    pass  # this is when there is no value for key
            if value is not None:
                setattr(self, key, value)

    def __is_key_input_image(self, key):
        """Check if a key of the App is an input parameter image list."""
        return self.app.GetParameterType(key) in (otb.ParameterType_InputImage, otb.ParameterType_InputImageList)

    def __is_key_list(self, key):
        """Check if a key of the App is an input parameter list."""
        return self.app.GetParameterType(key) in (otb.ParameterType_InputImageList, otb.ParameterType_StringList,
                                                  otb.ParameterType_InputFilenameList, otb.ParameterType_ListView,
                                                  otb.ParameterType_InputVectorDataList)

    def __is_key_images_list(self, key):
        """Check if a key of the App is an input parameter image list."""
        return self.app.GetParameterType(key) in (otb.ParameterType_InputImageList, otb.ParameterType_InputFilenameList)

    # Special methods
    def __str__(self):
        """Return a nice string representation with object id."""
        return f'<pyotb.App {self.appname} object id {id(self)}>'


class Slicer(App):
    """Slicer objects i.e. when we call something like raster[:, :, 2] from Python."""

    def __init__(self, x, rows, cols, channels):
        """Create a slicer object, that can be used directly for writing or inside a BandMath.

        It contains :
        - an ExtractROI app that handles extracting bands and ROI and can be written to disk or used in pipelines
        - in case the user only wants to extract one band, an expression such as "im1b#"

        Args:
            x: input
            rows: slice along Y / Latitude axis
            cols: slice along X / Longitude axis
            channels: channels, can be slicing, list or int

        """
        # Initialize the app that will be used for writing the slicer
        self.name = 'Slicer'

        self.output_parameter_key = 'out'
        parameters = {'in': x, 'mode': 'extent'}
        super().__init__('ExtractROI', parameters, preserve_dtype=True, frozen=True)
        # Channel slicing
        if channels != slice(None, None, None):
            # Trigger source app execution if needed
            nb_channels = get_nbchannels(x)
            self.app.Execute()  # this is needed by ExtractROI for setting the `cl` parameter
            # if needed, converting int to list
            if isinstance(channels, int):
                channels = [channels]
            # if needed, converting slice to list
            elif isinstance(channels, slice):
                channels_start = channels.start if channels.start is not None else 0
                channels_start = channels_start if channels_start >= 0 else nb_channels + channels_start
                channels_end = channels.stop if channels.stop is not None else nb_channels
                channels_end = channels_end if channels_end >= 0 else nb_channels + channels_end
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
        # TODO TBD: handle the step value in the slice so that NN undersampling is possible ? e.g. raster[::2, ::2]
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
        self.set_parameters(**parameters)
        self.execute()

        # These are some attributes when the user simply wants to extract *one* band to be used in an Operation
        if not spatial_slicing and isinstance(channels, list) and len(channels) == 1:
            self.one_band_sliced = channels[0] + 1  # OTB convention: channels start at 1
            self.input = x


class Input(App):
    """Class for transforming a filepath to pyOTB object."""

    def __init__(self, filepath):
        """Constructor for an Input object.

        Args:
            filepath: raster file path

        """
        self.filepath = filepath
        super().__init__('ExtractROI', {'in': self.filepath}, preserve_dtype=True)

    def __str__(self):
        """Return a nice string representation with input file path."""
        return f'<pyotb.Input object from {self.filepath}>'


class Output(otbObject):
    """Class for output of an app."""

    def __init__(self, app, output_parameter_key):
        """Constructor for an Output object.

        Args:
            app: The pyotb App
            output_parameter_key: Output parameter key

        """
        # Keeping the OTB app and the pyotb app
        self.pyotb_app, self.app = app, app.app
        self.parameters = self.pyotb_app.parameters
        self.output_param = output_parameter_key
        self.name = f'Output {output_parameter_key} from {self.app.GetName()}'

    def summarize(self):
        """Return the summary of the pipeline that generates the Output object.

        Returns:
            Nested dictionary summarizing the pipeline that generates the Output object.

        """
        return self.pyotb_app.summarize()

    def __str__(self):
        """Return a nice string representation with object id."""
        return f'<pyotb.Output {self.app.GetName()} object, id {id(self)}>'


class Operation(App):
    """Class for arithmetic/math operations done in Python.

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
        """Given some inputs and an operator, this function enables to transform this into an OTB application.

        Operations generally involve 2 inputs (+, -...). It can have only 1 input for `abs` operator.
        It can have 3 inputs for the ternary operator `cond ? x : y`.

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            *inputs: inputs. Can be App, Output, Input, Operation, Slicer, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        self.operator = operator
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
        self.output_param = 'out'

        # Computing the BandMath or BandMathX app
        self.exp_bands, self.exp = self.get_real_exp(self.fake_exp_bands)
        self.name = f'Operation exp="{self.exp}"'

        appname = 'BandMath' if len(self.exp_bands) == 1 else 'BandMathX'
        super().__init__(appname, il=self.unique_inputs, exp=self.exp)

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        """Create a 'fake' expression.

        E.g for the operation input1 + input2, we create a fake expression that is like "str(input1) + str(input2)"

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
            if any(isinstance(inp, Slicer) and hasattr(inp, 'one_band_sliced') for inp in inputs):
                nb_bands = 1
            else:
                nb_bands_list = [get_nbchannels(inp) for inp in inputs if not isinstance(inp, (float, int))]
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
        """Generates the BandMathX expression.

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
        """This an internal function, only to be used by `create_fake_exp`.

        Enable to create a fake expression just for one input and one band.

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
        """Return a nice string representation with object id."""
        return f'<pyotb.Operation `{self.operator}` object, id {id(self)}>'


class logicalOperation(Operation):
    """A specialization of Operation class for boolean logical operations i.e. >, <, >=, <=, ==, !=, `&` and `|`.

    The only difference is that not only the BandMath expression is saved (e.g. "im1b1 > 0 ? 1 : 0"), but also the
    logical expression (e.g. "im1b1 > 0")

    """

    def __init__(self, operator, *inputs, nb_bands=None):
        """Constructor for a logicalOperation object.

        Args:
            operator: string operator (one of >, <, >=, <=, ==, !=, &, |)
            *inputs: inputs
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        super().__init__(operator, *inputs, nb_bands=nb_bands)
        self.logical_exp_bands, self.logical_exp = self.get_real_exp(self.logical_fake_exp_bands)

    def create_fake_exp(self, operator, inputs, nb_bands=None):
        """Create a 'fake' expression.

        E.g for the operation input1 > input2, we create a fake expression that is like
        "str(input1) > str(input2) ? 1 : 0" and a logical fake expression that is like "str(input1) > str(input2)"

        Args:
            operator: str (one of >, <, >=, <=, ==, !=, &, |)
            inputs: Can be App, Output, Input, Operation, Slicer, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        # For any other operations, the output number of bands is the same as inputs
        if any(isinstance(inp, Slicer) and hasattr(inp, 'one_band_sliced') for inp in inputs):
            nb_bands = 1
        else:
            nb_bands_list = [get_nbchannels(inp) for inp in inputs if not isinstance(inp, (float, int))]
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
    """Get the nb of bands of input image.

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
            info = App("ReadImageInfo", inp, quiet=True)
            nb_channels = info.GetParameterInt("numberbands")
        except Exception as e:  # this happens when we pass a str that is not a filepath
            raise TypeError(f'Could not get the number of channels of `{inp}`. Not a filepath or wrong filepath') from e
    return nb_channels


def get_pixel_type(inp):
    """Get the encoding of input image pixels.

    Args:
        inp: can be filepath or pyotb object

    Returns:
        pixel_type: OTB enum e.g. `otbApplication.ImagePixelType_uint8', which actually is an int.
                    For an App with several outputs, only the pixel type of the first output is returned

    """
    if isinstance(inp, str):
        # Executing the app, without printing its log
        try:
            info = App("ReadImageInfo", inp, quiet=True)
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
    """Convert one str pixel type to OTB integer enum if necessary.

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
