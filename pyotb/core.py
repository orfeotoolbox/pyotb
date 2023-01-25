# -*- coding: utf-8 -*-
"""This module is the core of pyotb."""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import otbApplication as otb  # pylint: disable=import-error

from .helpers import logger


class OTBObject:
    def __init__(self, name: str, app: otb.Application, image_dic: dict = None):
        """
            name:
            app:
            image_dic: enables to keep a reference to image_dic. image_dic is a dictionary, such as
                       the result of app.ExportImage(). Use it when the app takes a numpy array as input.
                       See this related issue for why it is necessary to keep reference of object:
                       https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/1824


        """
        self.name = name
        self.app = app
        self.image_dic = image_dic

        self.parameters_keys = tuple(self.app.GetParametersKeys())
        self.all_param_types = {k: self.app.GetParameterType(k) for k in self.parameters_keys}
        self.out_param_types = {k: v for k, v in self.all_param_types.items()
                                if v in (otb.ParameterType_OutputImage,
                                         otb.ParameterType_OutputVectorData,
                                         otb.ParameterType_OutputFilename)}

        self.transform = None  # TODO: value?
        self.exports_dic = {}

    def get_first_key(self, param_types: list[str]) -> str:
        """Get the first output param key for specific file types."""
        for key, param_type in sorted(self.all_param_types.items()):
            if param_type in param_types:
                return key
        return None

    @property
    def key_input(self) -> str:
        """Get the name of first input parameter, raster > vector > file."""
        return self.get_first_key(param_types=[otb.ParameterType_InputImage,
                                               otb.ParameterType_InputImageList]) \
            or self.get_first_key(param_types=[otb.ParameterType_InputVectorData,
                                               otb.ParameterType_InputVectorDataList]) \
            or self.get_first_key(param_types=[otb.ParameterType_InputFilename,
                                               otb.ParameterType_InputFilenameList])

    @property
    def key_input_image(self) -> str:
        """Get the name of first input image parameter."""
        return self.get_first_key(param_types=[otb.ParameterType_InputImage, otb.ParameterType_InputImageList])

    @property
    def key_output_image(self) -> str:
        """Get the name of first output image parameter."""
        return self.get_first_key(param_types=[otb.ParameterType_OutputImage])

    @property
    def metadata(self):
        """Return first output image metadata dictionary."""
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        return dict(self.app.GetMetadataDictionary(self.key_output_image))

    @property
    def dtype(self) -> np.dtype:
        """Expose the pixel type of output image using numpy convention.

        Returns:
            dtype: pixel type of the output image

        """
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        enum = self.app.GetParameterOutputImagePixelType(self.key_output_image)
        return self.app.ConvertPixelTypeToNumpy(enum)

    @property
    def shape(self) -> tuple(int):
        """Enables to retrieve the shape of a pyotb object using numpy convention.

        Returns:
            shape: (height, width, bands)

        """
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        width, height = self.app.GetImageSize(self.key_output_image)
        bands = self.app.GetImageNbBands(self.key_output_image)
        return height, width, bands

    @property
    def transform(self) -> tuple(int):
        """Get image affine transform, rasterio style (see https://www.perrygeo.com/python-affine-transforms.html).

        Returns:
            transform: (X spacing, X offset, X origin, Y offset, Y spacing, Y origin)
        """
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        spacing_x, spacing_y = self.app.GetImageSpacing(self.key_output_image)
        origin_x, origin_y = self.app.GetImageOrigin(self.key_output_image)
        # Shift image origin since OTB is giving coordinates of pixel center instead of corners
        origin_x, origin_y = origin_x - spacing_x / 2, origin_y - spacing_y / 2
        return spacing_x, 0.0, origin_x, 0.0, spacing_y, origin_y

    def get_infos(self):
        """Return a dict output of ReadImageInfo for the first image output."""
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        return OTBObject("ReadImageInfo", self, quiet=True).data

    def get_statistics(self):
        """Return a dict output of ComputeImagesStatistics for the first image output."""
        if not self.key_output_image:
            raise TypeError(f"{self.name}: this application has no raster output")
        return OTBObject("ComputeImagesStatistics", self, quiet=True).data

    def read_values_at_coords(self, row: int, col: int, bands: int = None) -> list[int | float] | int | float:
        """Get pixel value(s) at a given YX coordinates.

        Args:
            row: index along Y / latitude axis
            col: index along X / longitude axis
            bands: band number, list or slice to fetch values from

        Returns:
            single numerical value or a list of values for each band

        """
        channels = []
        app = OTBObject("PixelValue", self, coordx=col, coordy=row, frozen=True, quiet=True)
        if bands is not None:
            if isinstance(bands, int):
                if bands < 0:
                    bands = self.shape[2] + bands
                channels = [bands]
            elif isinstance(bands, slice):
                channels = self.channels_list_from_slice(bands)
            elif not isinstance(bands, list):
                raise TypeError(f"{self.name}: type '{type(bands)}' cannot be interpreted as a valid slicing")
            if channels:
                app.app.Execute()
                app.set_parameters({"cl": [f"Channel{n + 1}" for n in channels]})
        app.execute()
        data = literal_eval(app.app.GetParameterString("value"))
        if len(channels) == 1:
            return data[0]
        return data

    def channels_list_from_slice(self, bands: int) -> list[int]:
        """Get list of channels to read values at, from a slice."""
        nb_channels = self.shape[2]
        start, stop, step = bands.start, bands.stop, bands.step
        start = nb_channels + start if isinstance(start, int) and start < 0 else start
        stop = nb_channels + stop if isinstance(stop, int) and stop < 0 else stop
        step = 1 if step is None else step
        if start is not None and stop is not None:
            return list(range(start, stop, step))
        if start is not None and stop is None:
            return list(range(start, nb_channels, step))
        if start is None and stop is not None:
            return list(range(0, stop, step))
        if start is None and stop is None:
            return list(range(0, nb_channels, step))
        raise ValueError(f"{self.name}: '{bands}' cannot be interpreted as valid slicing.")

    def summarize(self) -> dict:
        """Serialize an object and its pipeline into a dictionary.

        Returns:
            nested dictionary summarizing the pipeline

        """
        params = self.parameters
        for k, p in params.items():
            # In the following, we replace each parameter which is an OTBObject, with its summary.
            if isinstance(p, OTBObject):  # single parameter
                params[k] = p.summarize()
            elif isinstance(p, list):  # parameter list
                params[k] = [pi.summarize() if isinstance(pi, OTBObject) else pi for pi in p]
        return {"name": self.name, "parameters": params}

    def export(self, key: str = None, preserve_dtype: bool = True) -> dict[str, dict[str, np.ndarray]]:
        """Export a specific output image as numpy array and store it in object exports_dic.

        Args:
            key: parameter key to export, if None then the default one will be used
            preserve_dtype: when set to True, the numpy array is converted to the same pixel type as
                            the OTBObject first output. Default is True

        Returns:
            the exported numpy array

        """
        if key is None:
            key = self.key_output_image
        if key not in self.exports_dic:
            self.exports_dic[key] = self.app.ExportImage(key)
        if preserve_dtype:
            self.exports_dic[key]["array"] = self.exports_dic[key]["array"].astype(self.dtype)
        return self.exports_dic[key]

    def to_numpy(self, key: str = None, preserve_dtype: bool = True, copy: bool = False) -> np.ndarray:
        """Export a pyotb object to numpy array.

        Args:
            key: the output parameter name to export as numpy array
            preserve_dtype: when set to True, the numpy array is converted to the same pixel type as
                            the OTBObject first output. Default is True
            copy: whether to copy the output array, default is False
                  required to True if preserve_dtype is False and the source app reference is lost

        Returns:
            a numpy array

        """
        data = self.export(key, preserve_dtype)
        array = data["array"]
        if copy:
            return array.copy()
        return array

    def to_rasterio(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Export image as a numpy array and its metadata compatible with rasterio.

        Returns:
          array : a numpy array in the (bands, height, width) order
          profile: a metadata dict required to write image using rasterio

        """
        array = self.to_numpy(preserve_dtype=True, copy=False)
        array = np.moveaxis(array, 2, 0)
        proj = self.app.GetImageProjection(self.key_output_image)
        profile = {
            'crs': proj, 'dtype': array.dtype,
            'count': array.shape[0], 'height': array.shape[1], 'width': array.shape[2],
            'transform': self.transform
        }
        return array, profile

    def xy_to_rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Find (row, col) index using (x, y) projected coordinates - image CRS is expected.

        Args:
            x: longitude or projected X
            y: latitude or projected Y

        Returns:
            pixel index: (row, col)
        """
        spacing_x, _, origin_x, _, spacing_y, origin_y = self.transform
        row, col = (origin_y - y) / spacing_y, (x - origin_x) / spacing_x
        return abs(int(row)), int(col)

    # Special functions
    def __hash__(self):
        """Override the default behaviour of the hash function.

        Returns:
            self hash

        """
        return id(self)

    def __getattr__(self, name):
        """This method is called when the default attribute access fails.

        We choose to access the attribute `name` of self.app.
        Thus, any method of otbApplication can be used transparently on OTBObject objects,
        e.g. SetParameterOutputImagePixelType() or ExportImage() work

        Args:
            name: attribute name

        Returns:
            attribute

        Raises:
            AttributeError: when `name` is not an attribute of self.app

        """
        if name in dir(self.app):
            return getattr(self.app, name)
        raise AttributeError(f"{self.name}: could not find attribute `{name}`")

    def __getitem__(self, key):
        """Override the default __getitem__ behaviour.

        This function enables 2 things :
        - access attributes like that : object['any_attribute']
        - slicing, i.e. selecting ROI/bands. For example, selecting first 3 bands: object[:, :, :3]
                                                          selecting bands 1, 2 & 5 : object[:, :, [0, 1, 4]]
                                                          selecting 1000x1000 subset : object[:1000, :1000]
        - access pixel value(s) at a specified row, col index

        Args:
            key: attribute key

        Returns:
            attribute, pixel values or Slicer

        """
        # Accessing string attributes
        if isinstance(key, str):
            return self.__dict__.get(key)
        # Accessing pixel value(s) using Y/X coordinates
        if isinstance(key, tuple) and len(key) >= 2:
            row, col = key[0], key[1]
            if isinstance(row, int) and isinstance(col, int):
                if row < 0 or col < 0:
                    raise ValueError(f"{self.name}: can't read pixel value at negative coordinates ({row}, {col})")
                channels = None
                if len(key) == 3:
                    channels = key[2]
                return self.read_values_at_coords(row, col, channels)
        # Slicing
        if not isinstance(key, tuple) or (isinstance(key, tuple) and (len(key) < 2 or len(key) > 3)):
            raise ValueError(f'"{key}"cannot be interpreted as valid slicing. Slicing should be 2D or 3D.')
        if isinstance(key, tuple) and len(key) == 2:
            # Adding a 3rd dimension
            key = key + (slice(None, None, None),)
        return Slicer(self, *key)

    def __add__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default addition and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self + other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("+", self, other)

    def __sub__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self - other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("-", self, other)

    def __mul__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self * other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("*", self, other)

    def __truediv__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self / other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("/", self, other)

    def __radd__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default reverse addition and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            other + self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("+", other, self)

    def __rsub__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default subtraction and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            other - self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("-", other, self)

    def __rmul__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default multiplication and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            other * self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("*", other, self)

    def __rtruediv__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default division and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            other / self

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return Operation("/", other, self)

    def __abs__(self) -> Operation:
        """Overrides the default abs operator and flavours it with BandMathX.

        Returns:
            abs(self)

        """
        return Operation("abs", self)

    def __ge__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default greater or equal and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self >= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation(">=", self, other)

    def __le__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default less or equal and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self <= other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("<=", self, other)

    def __gt__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default greater operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self > other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation(">", self, other)

    def __lt__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default less operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self < other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("<", self, other)

    def __eq__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default eq operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self == other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("==", self, other)

    def __ne__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default different operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self != other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("!=", self, other)

    def __or__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default or operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self || other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("||", self, other)

    def __and__(self, other: OTBObject | str | int | float) -> Operation:
        """Overrides the default and operator and flavours it with BandMathX.

        Args:
            other: the other member of the operation

        Returns:
            self && other

        """
        if isinstance(other, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return LogicalOperation("&&", self, other)

    # TODO: other operations ?
    #  e.g. __pow__... cf https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __array__(self) -> np.ndarray:
        """This is called when running np.asarray(pyotb_object).

        Returns:
            a numpy array

        """
        return self.to_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> OTBObject:
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
        if method == "__call__":
            # Converting potential pyotb inputs to arrays
            arrays = []
            image_dic = None
            for inp in inputs:
                if isinstance(inp, (float, int, np.ndarray, np.generic)):
                    arrays.append(inp)
                elif isinstance(inp, OTBObject):
                    if not inp.exports_dic:
                        inp.export()
                    image_dic = inp.exports_dic[inp.key_output_image]
                    array = image_dic["array"]
                    arrays.append(array)
                else:
                    logger.debug(type(self))
                    return NotImplemented
            # Performing the numpy operation
            result_array = ufunc(*arrays, **kwargs)
            result_dic = image_dic
            result_dic["array"] = result_array
            # Importing back to OTB, pass the result_dic just to keep reference
            app = OTBObject("ExtractROI", image_dic=result_dic, frozen=True, quiet=True)
            if result_array.shape[2] == 1:
                app.ImportImage("in", result_dic)
            else:
                app.ImportVectorImage("in", result_dic)
            app.execute()
            return app
        return NotImplemented


class App(OTBObject):
    """Base class that gathers common operations for any OTB application."""

    def __init__(self, otb_app_name: str, *args, frozen: bool = False, quiet: bool = False, image_dic: dict = None,
                 **kwargs):
        """Common constructor for OTB applications. Handles in-memory connection between apps.

        Args:
            otb_app_name: name of the OTB application, e.g. 'BandMath'
            *args: used for passing application parameters. Can be :
                           - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                             (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                           - string or OTBObject, useful when the user wants to specify the input "in"
                           - list, useful when the user wants to specify the input list 'il'
            frozen: freeze OTB app in order to use execute() later and avoid blocking process during __init___
            quiet: whether to print logs of the OTB app
            image_dic: enables to keep a reference to image_dic. image_dic is a dictionary, such as
                       the result of app.ExportImage(). Use it when the app takes a numpy array as input.
                       See this related issue for why it is necessary to keep reference of object:
                       https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/1824
            **kwargs: used for passing application parameters.
                      e.g. il=['input1.tif', App_object2, App_object3.out], out='output.tif'

        """

        self.frozen = frozen
        self.quiet = quiet
        create = otb.Registry.CreateApplicationWithoutLogger if quiet else otb.Registry.CreateApplication
        super().__init__(name=f"OTB Application {otb_app_name}", app=create(otb_app_name), image_dic=image_dic)

        # Set parameters
        self.parameters = {}
        if args or kwargs:
            self.set_parameters(*args, **kwargs)
        if not self.frozen:
            self.execute()
            if any(key in self.parameters for key in self.out_param_types):
                self.flush()  # auto flush if any output param was provided during app init

        # Elapsed time
        self.time_start, self.time_end = 0, 0

    @property
    def data(self):
        """Expose app's output data values in a dictionary."""
        skip_keys = ("ram", "elev.default", "mapproj.utm.zone", "mapproj.utm.northhem")
        skip_keys = skip_keys + tuple(self.out_param_types) + tuple(self.parameters)
        keys = (k for k in self.parameters_keys if k not in skip_keys)

        def _check(v):
            return not isinstance(v, otb.ApplicationProxy) and v not in ("", None, [], ())

        return {str(k): self[k] for k in keys if _check(self[k])}

    def set_parameters(self, *args, **kwargs):
        """Set some parameters of the app.

        When useful, e.g. for images list, this function appends the parameters
        instead of overwriting them. Handles any parameters, i.e. in-memory & filepaths

        Args:
            *args: Can be : - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                              (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                            - string or OTBObject, useful when the user implicitly wants to set the param "in"
                            - list, useful when the user implicitly wants to set the param "il"
            **kwargs: keyword arguments e.g. il=['input1.tif', oApp_object2, App_object3.out], out='output.tif'

        Raises:
            Exception: when the setting of a parameter failed

        """
        parameters = kwargs
        parameters.update(self.__parse_args(args))
        # Going through all arguments
        for key, obj in parameters.items():
            if key not in self.parameters_keys:
                raise KeyError(f'{self.name}: unknown parameter name "{key}"')
            # When the parameter expects a list, if needed, change the value to list
            if is_key_list(self, key) and not isinstance(obj, (list, tuple)):
                obj = [obj]
                logger.info('%s: argument for parameter "%s" was converted to list', self.name, key)
            try:
                # This is when we actually call self.app.SetParameter*
                self.__set_param(key, obj)
            except (RuntimeError, TypeError, ValueError, KeyError) as e:
                raise Exception(
                    f"{self.name}: something went wrong before execution "
                    f"(while setting parameter '{key}' to '{obj}')"
                ) from e
        # Update _parameters using values from OtbApplication object
        otb_params = self.app.GetParameters().items()
        otb_params = {k: str(v) if isinstance(v, otb.ApplicationProxy) else v for k, v in otb_params}
        # Update param dict and save values as object attributes
        self.parameters.update({**parameters, **otb_params})
        self.save_objects()

    def save_objects(self):
        """Saving app parameters and outputs as attributes, so that they can be accessed with `obj.key`.

        This is useful when the key contains reserved characters such as a point eg "io.out"
        """
        for key in self.parameters_keys:
            if key in dir(self.__class__):
                continue  # skip forbidden attribute since it is already used by the class
            value = self.parameters.get(key)  # basic parameters
            if value is None:
                try:
                    value = self.app.GetParameterValue(key)  # any other app attribute (e.g. ReadImageInfo results)
                except RuntimeError:
                    continue  # this is when there is no value for key
            # Convert output param path to Output object
            if key in self.out_param_types:
                value = Output(self, key, value)
            elif isinstance(value, str):
                try:
                    value = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
            # Save attribute
            setattr(self, key, value)

    def execute(self):
        """Execute and write to disk if any output parameter has been set during init."""
        logger.debug("%s: run execute() with parameters=%s", self.name, self.parameters)
        self.time_start = perf_counter()
        try:
            self.app.Execute()
        except (RuntimeError, FileNotFoundError) as e:
            raise Exception(f"{self.name}: error during during app execution") from e
        self.frozen = False
        self.time_end = perf_counter()
        logger.debug("%s: execution ended", self.name)
        self.save_objects()  # this is required for apps like ReadImageInfo or ComputeImagesStatistics

    def flush(self):
        """Flush data to disk, this is when WriteOutput is actually called."""
        try:
            logger.debug("%s: flushing data to disk", self.name)
            self.app.WriteOutput()
        except RuntimeError:
            logger.debug("%s: failed with WriteOutput, executing once again with ExecuteAndWriteOutput", self.name)
            self.app.ExecuteAndWriteOutput()
        self.time_end = perf_counter()

    def write(self, *args, filename_extension: str = "", pixel_type: dict[str, str] | str = None,
              preserve_dtype: bool = False, **kwargs):
        """Set output pixel type and write the output raster files.

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
            preserve_dtype: propagate main input pixel type to outputs, in case pixel_type is None
            **kwargs: keyword arguments e.g. out='output.tif'

        """
        # Gather all input arguments in kwargs dict
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, str) and kwargs:
                logger.warning('%s: keyword arguments specified, ignoring argument "%s"', self.name, arg)
            elif isinstance(arg, str) and self.key_output_image:
                kwargs.update({self.key_output_image: arg})

        # Append filename extension to filenames
        if filename_extension:
            logger.debug("%s: using extended filename for outputs: %s", self.name, filename_extension)
            if not filename_extension.startswith("?"):
                filename_extension = "?" + filename_extension
            for key, value in kwargs.items():
                if self.out_param_types[key] == otb.ParameterType_OutputImage and '?' not in value:
                    kwargs[key] = value + filename_extension

        # Manage output pixel types
        dtypes = {}
        if pixel_type:
            if isinstance(pixel_type, str):
                type_name = self.app.ConvertPixelTypeToNumpy(parse_pixel_type(pixel_type))
                logger.debug('%s: output(s) will be written with type "%s"', self.name, type_name)
                for key in kwargs:
                    if self.out_param_types[key] == otb.ParameterType_OutputImage:
                        dtypes[key] = parse_pixel_type(pixel_type)
            elif isinstance(pixel_type, dict):
                dtypes = {k: parse_pixel_type(v) for k, v in pixel_type.items()}
        elif preserve_dtype:
            self.propagate_dtype()  # all outputs will have the same type as the main input raster

        # Set parameters and flush to disk
        for key, output_filename in kwargs.items():
            if key in dtypes:
                self.propagate_dtype(key, dtypes[key])
            self.set_parameters({key: output_filename})
        self.flush()

    def propagate_dtype(self, target_key: str = None, dtype: int = None):
        """Propagate a pixel type from main input to every outputs, or to a target output key only.

        With multiple inputs (if dtype is not provided), the type of the first input is considered.
        With multiple outputs (if target_key is not provided), all outputs will be converted to the same pixel type.

        Args:
            target_key: output param key to change pixel type
            dtype: data type to use

        """
        if not dtype:
            param = self.parameters.get(self.key_input_image)
            if not param:
                logger.warning("%s: could not propagate pixel type from inputs to output", self.name)
                return
            if isinstance(param, (list, tuple)):
                param = param[0]  # first image in "il"
            try:
                dtype = get_pixel_type(param)
            except (TypeError, RuntimeError):
                logger.warning('%s: unable to identify pixel type of key "%s"', self.name, param)
                return
        if target_key:
            keys = [target_key]
        else:
            keys = [k for k, v in self.out_param_types.items() if v == otb.ParameterType_OutputImage]
        for key in keys:
            self.app.SetParameterOutputImagePixelType(key, dtype)

    # Private functions
    def __parse_args(self, args: list[str | OTBObject | dict | list]) -> dict[str, Any]:
        """Gather all input arguments in kwargs dict.

        Args:
            args: the list of arguments passed to set_parameters()

        Returns:
            a dictionary with the right keyword depending on the object

        """
        kwargs = {}
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            elif isinstance(arg, (str, OTBObject)) or isinstance(arg, list) and is_key_list(self, self.key_input):
                kwargs.update({self.key_input: arg})
        return kwargs

    def __set_param(self, key: str, obj: list | tuple | OTBObject | otb.Application | list[Any]):
        """Set one parameter, decide which otb.Application method to use depending on target object."""
        if obj is None or (isinstance(obj, (list, tuple)) and not obj):
            self.app.ClearValue(key)
            return
        if key not in self.parameters_keys:
            raise KeyError(
                f"{self.name}: parameter '{key}' was not recognized. Available keys are {self.parameters_keys}"
            )
        # Single-parameter cases
        if isinstance(obj, OTBObject):
            self.app.ConnectImage(key, obj.app, obj.key_output_image)
        elif isinstance(obj, otb.Application):  # this is for backward comp with plain OTB
            self.app.ConnectImage(key, obj, get_out_images_param_keys(obj)[0])
        elif key == "ram":  # SetParameterValue in OTB<7.4 doesn't work for ram parameter cf gitlab OTB issue 2200
            self.app.SetParameterInt("ram", int(obj))
        elif not isinstance(obj, list):  # any other parameters (str, int...)
            self.app.SetParameterValue(key, obj)
        # Images list
        elif is_key_images_list(self, key):
            # To enable possible in-memory connections, we go through the list and set the parameters one by one
            for inp in obj:
                if isinstance(inp, OTBObject):
                    self.app.ConnectImage(key, inp.app, inp.key_output_image)
                elif isinstance(inp, otb.Application):  # this is for backward comp with plain OTB
                    self.app.ConnectImage(key, obj, get_out_images_param_keys(inp)[0])
                else:  # here `input` should be an image filepath
                    # Append `input` to the list, do not overwrite any previously set element of the image list
                    self.app.AddParameterStringList(key, inp)
        # List of any other types (str, int...)
        else:
            self.app.SetParameterValue(key, obj)

    def __str__(self):
        """Return a nice string representation with object id."""
        return f"<pyotb.App {self.name} object id {id(self)}>"


class Slicer(App):
    """Slicer objects i.e. when we call something like raster[:, :, 2] from Python."""

    def __init__(self, obj: OTBObject | str, rows: int, cols: int, channels: int):
        """Create a slicer object, that can be used directly for writing or inside a BandMath.

        It contains :
        - an ExtractROI app that handles extracting bands and ROI and can be written to disk or used in pipelines
        - in case the user only wants to extract one band, an expression such as "im1b#"

        Args:
            obj: input
            rows: slice along Y / Latitude axis
            cols: slice along X / Longitude axis
            channels: channels, can be slicing, list or int

        """
        super().__init__("ExtractROI", {"in": obj, "mode": "extent"}, quiet=True, frozen=True)
        self.name = "Slicer"
        self.rows, self.cols = rows, cols
        parameters = {}

        # Channel slicing
        if channels != slice(None, None, None):
            # Trigger source app execution if needed
            nb_channels = get_nbchannels(obj)
            self.app.Execute()  # this is needed by ExtractROI for setting the `cl` parameter
            # if needed, converting int to list
            if isinstance(channels, int):
                channels = [channels]
            # if needed, converting slice to list
            elif isinstance(channels, slice):
                channels = self.channels_list_from_slice(channels)
            elif isinstance(channels, tuple):
                channels = list(channels)
            elif not isinstance(channels, list):
                raise ValueError(f"Invalid type for channels, should be int, slice or list of bands. : {channels}")
            # Change the potential negative index values to reverse index
            channels = [c if c >= 0 else nb_channels + c for c in channels]
            parameters.update({"cl": [f"Channel{i + 1}" for i in channels]})

        # Spatial slicing
        spatial_slicing = False
        # TODO TBD: handle the step value in the slice so that NN undersampling is possible ? e.g. raster[::2, ::2]
        if rows.start is not None:
            parameters.update({"mode.extent.uly": rows.start})
            spatial_slicing = True
        if rows.stop is not None and rows.stop != -1:
            parameters.update({"mode.extent.lry": rows.stop - 1})  # subtract 1 to respect python convention
            spatial_slicing = True
        if cols.start is not None:
            parameters.update({"mode.extent.ulx": cols.start})
            spatial_slicing = True
        if cols.stop is not None and cols.stop != -1:
            parameters.update({"mode.extent.lrx": cols.stop - 1})  # subtract 1 to respect python convention
            spatial_slicing = True
        # These are some attributes when the user simply wants to extract *one* band to be used in an Operation
        if not spatial_slicing and isinstance(channels, list) and len(channels) == 1:
            self.one_band_sliced = channels[0] + 1  # OTB convention: channels start at 1
            self.input = obj

        # Execute app
        self.set_parameters(parameters)
        self.propagate_dtype()
        self.execute()


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

    def __init__(self, operator: str, *inputs, nb_bands: int = None):
        """Given some inputs and an operator, this function enables to transform this into an OTB application.

        Operations generally involve 2 inputs (+, -...). It can have only 1 input for `abs` operator.
        It can have 3 inputs for the ternary operator `cond ? x : y`.

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            *inputs: inputs. Can be OTBObject, filepath, int or float
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
                    self.im_dic[str(inp)] = f"im{self.im_count}"
                    mapping_str_to_input[str(inp)] = inp
                    self.im_count += 1
        # Getting unique image inputs, in the order im1, im2, im3 ...
        self.unique_inputs = [mapping_str_to_input[str_input] for str_input in sorted(self.im_dic, key=self.im_dic.get)]
        self.exp_bands, self.exp = self.get_real_exp(self.fake_exp_bands)
        # Execute app
        name = "BandMath" if len(self.exp_bands) == 1 else "BandMathX"
        super().__init__(name, il=self.unique_inputs, exp=self.exp, quiet=True)
        self.name = f'Operation exp="{self.exp}"'

    def create_fake_exp(self, operator: str, inputs: list[OTBObject | str | int | float],
                        nb_bands: int = None):
        """Create a 'fake' expression.

        E.g for the operation input1 + input2, we create a fake expression that is like "str(input1) + str(input2)"

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            inputs: inputs. Can be OTBObject, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        self.inputs.clear()
        self.nb_channels.clear()
        logger.debug("%s, %s", operator, inputs)
        # This is when we use the ternary operator with `pyotb.where` function. The output nb of bands is already known
        if operator == "?" and nb_bands:
            pass
        # For any other operations, the output number of bands is the same as inputs
        else:
            if any(isinstance(inp, Slicer) and hasattr(inp, "one_band_sliced") for inp in inputs):
                nb_bands = 1
            else:
                nb_bands_list = [get_nbchannels(inp) for inp in inputs if not isinstance(inp, (float, int))]
                # check that all inputs have the same nb of bands
                if len(nb_bands_list) > 1:
                    if not all(x == nb_bands_list[0] for x in nb_bands_list):
                        raise ValueError("All images do not have the same number of bands")
                nb_bands = nb_bands_list[0]

        # Create a list of fake expressions, each item of the list corresponding to one band
        self.fake_exp_bands.clear()
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for k, inp in enumerate(inputs):
                # Generating the fake expression of the current input,
                # this is a special case for the condition of the ternary operator `cond ? x : y`
                if len(inputs) == 3 and k == 0:
                    # When cond is monoband whereas the result is multiband, we expand the cond to multiband
                    if nb_bands != inp.shape[2]:
                        cond_band = 1
                    else:
                        cond_band = band
                    fake_exp, corresponding_inputs, nb_channels = self.create_one_input_fake_exp(
                        inp, cond_band, keep_logical=True
                    )
                else:
                    # Any other input
                    fake_exp, corresponding_inputs, nb_channels = self.create_one_input_fake_exp(
                        inp, band, keep_logical=False
                    )
                fake_exps.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresponding_inputs and nb_channels:
                    self.inputs.extend(corresponding_inputs)
                    self.nb_channels.update(nb_channels)

            # Generating the fake expression of the whole operation
            if len(inputs) == 1:  # this is only for 'abs'
                fake_exp = f"({operator}({fake_exps[0]}))"
            elif len(inputs) == 2:
                # We create here the "fake" expression. For example, for a BandMathX expression such as '2 * im1 + im2',
                # the false expression stores the expression 2 * str(input1) + str(input2)
                fake_exp = f"({fake_exps[0]} {operator} {fake_exps[1]})"
            elif len(inputs) == 3 and operator == "?":  # this is only for ternary expression
                fake_exp = f"({fake_exps[0]} ? {fake_exps[1]} : {fake_exps[2]})"
            self.fake_exp_bands.append(fake_exp)

    def get_real_exp(self, fake_exp_bands: str) -> tuple(list[str], str):
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
                # Replace the name of in-memory object (e.g. '<pyotb.App object>b1' by 'im1b1')
                one_band_exp = one_band_exp.replace(str(inp), self.im_dic[str(inp)])
            exp_bands.append(one_band_exp)
        # Form the final expression (e.g. 'im1b1 + 1; im1b2 + 1')
        exp = ";".join(exp_bands)
        return exp_bands, exp

    @staticmethod
    def create_one_input_fake_exp(x: OTBObject | str,
                                  band: int, keep_logical: bool = False) -> tuple(str, list[OTBObject], int):
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
        if isinstance(x, Slicer) and hasattr(x, "one_band_sliced"):
            if keep_logical and isinstance(x.input, LogicalOperation):
                fake_exp = x.input.logical_fake_exp_bands[x.one_band_sliced - 1]
                inputs = x.input.inputs
                nb_channels = x.input.nb_channels
            elif isinstance(x.input, Operation):
                # Keep only one band of the expression
                fake_exp = x.input.fake_exp_bands[x.one_band_sliced - 1]
                inputs = x.input.inputs
                nb_channels = x.input.nb_channels
            else:
                # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
                fake_exp = f"{x.input}b{x.one_band_sliced}"
                inputs = [x.input]
                nb_channels = {x.input: 1}
        # For LogicalOperation, we save almost the same attributes as an Operation
        elif keep_logical and isinstance(x, LogicalOperation):
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
            fake_exp = f"{x}b{band}"
        return fake_exp, inputs, nb_channels

    def __str__(self) -> str:
        """Return a nice string representation with operator and object id."""
        return f"<pyotb.Operation `{self.operator}` object, id {id(self)}>"


class LogicalOperation(Operation):
    """A specialization of Operation class for boolean logical operations i.e. >, <, >=, <=, ==, !=, `&` and `|`.

    The only difference is that not only the BandMath expression is saved (e.g. "im1b1 > 0 ? 1 : 0"), but also the
    logical expression (e.g. "im1b1 > 0")

    """

    def __init__(self, operator: str, *inputs, nb_bands: int = None):
        """Constructor for a LogicalOperation object.

        Args:
            operator: string operator (one of >, <, >=, <=, ==, !=, &, |)
            *inputs: inputs
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        super().__init__(operator, *inputs, nb_bands=nb_bands)
        self.logical_exp_bands, self.logical_exp = self.get_real_exp(self.logical_fake_exp_bands)

    def create_fake_exp(self, operator: str, inputs: list[OTBObject | str | int | float],
                        nb_bands: int = None):
        """Create a 'fake' expression.

        E.g for the operation input1 > input2, we create a fake expression that is like
        "str(input1) > str(input2) ? 1 : 0" and a logical fake expression that is like "str(input1) > str(input2)"

        Args:
            operator: str (one of >, <, >=, <=, ==, !=, &, |)
            inputs: Can be OTBObject, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        # For any other operations, the output number of bands is the same as inputs
        if any(isinstance(inp, Slicer) and hasattr(inp, "one_band_sliced") for inp in inputs):
            nb_bands = 1
        else:
            nb_bands_list = [get_nbchannels(inp) for inp in inputs if not isinstance(inp, (float, int))]
            # check that all inputs have the same nb of bands
            if len(nb_bands_list) > 1:
                if not all(x == nb_bands_list[0] for x in nb_bands_list):
                    raise ValueError("All images do not have the same number of bands")
            nb_bands = nb_bands_list[0]

        # Create a list of fake exp, each item of the list corresponding to one band
        for i, band in enumerate(range(1, nb_bands + 1)):
            fake_exps = []
            for inp in inputs:
                fake_exp, corresp_inputs, nb_channels = super().create_one_input_fake_exp(inp, band, keep_logical=True)
                fake_exps.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresp_inputs and nb_channels:
                    self.inputs.extend(corresp_inputs)
                    self.nb_channels.update(nb_channels)
            # We create here the "fake" expression. For example, for a BandMathX expression such as 'im1 > im2',
            # the logical fake expression stores the expression "str(input1) > str(input2)"
            logical_fake_exp = f"({fake_exps[0]} {operator} {fake_exps[1]})"
            # We keep the logical expression, useful if later combined with other logical operations
            self.logical_fake_exp_bands.append(logical_fake_exp)
            # We create a valid BandMath expression, e.g. "str(input1) > str(input2) ? 1 : 0"
            fake_exp = f"({logical_fake_exp} ? 1 : 0)"
            self.fake_exp_bands.append(fake_exp)


class Input(App):
    """Class for transforming a filepath to pyOTB object."""

    def __init__(self, path: str):
        """Default constructor.

        Args:
            path: Anything supported by GDAL (local file on the filesystem, remote resource e.g. /vsicurl/.., etc.)

        """
        self.path = path
        super().__init__("ExtractROI", {"in": path}, frozen=True)
        self.name = f"Input from {path}"
        self.propagate_dtype()
        self.execute()

    def __str__(self) -> str:
        """Return a nice string representation with file path."""
        return f"<pyotb.Input object from {self.path}>"


class Output(OTBObject):
    """Object that behave like a pointer to a specific application output file."""

    def __init__(self, pyotb_app: OTBObject,  # pylint: disable=super-init-not-called
                 param_key: str, filepath: str = None, mkdir: bool = True):
        """Constructor for an Output object.

        Args:
            pyotb_app: The pyotb App to store reference from
            param_key: Output parameter key of the target app
            filepath: path of the output file (if not in memory)
            mkdir: create missing parent directories

        """
        super().__init__(name=f"Output {param_key} from {self.pyotb_app.name}", app=pyotb_app)
        self.filepath = None
        if filepath:
            if '?' in filepath:
                filepath = filepath.split('?')[0]
            self.filepath = Path(filepath)
            if mkdir:
                self.make_parent_dirs()

    @property
    def key_output_image(self):
        """Overwrite OTBObject prop, in order to use Operation special methods with the right Output param_key."""
        return self.param_key

    def exists(self) -> bool:
        """Check file exist."""
        return self.filepath.exists()

    def make_parent_dirs(self):
        """Create missing parent directories."""
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True)

    def __str__(self) -> str:
        """Return a nice string representation with source app name and object id."""
        return f"<pyotb.Output {self.source_app.name} object, id {id(self)}>"


def get_nbchannels(inp: str | OTBObject) -> int:
    """Get the nb of bands of input image.

    Args:
        inp: can be filepath or pyotb object

    Returns:
        number of bands in image

    """
    if isinstance(inp, OTBObject):
        nb_channels = inp.shape[-1]
    else:
        # Executing the app, without printing its log
        try:
            info = OTBObject("ReadImageInfo", inp, quiet=True)
            nb_channels = info.GetParameterInt("numberbands")
        except Exception as e:  # this happens when we pass a str that is not a filepath
            raise TypeError(f'Could not get the number of channels of `{inp}`. Not a filepath or wrong filepath') from e
    return nb_channels


def get_pixel_type(inp: str | OTBObject) -> str:
    """Get the encoding of input image pixels.

    Args:
        inp: can be filepath or pyotb object

    Returns:
        pixel_type: OTB enum e.g. `otbApplication.ImagePixelType_uint8', which actually is an int.
                    For an App with several outputs, only the pixel type of the first output is returned

    """
    if isinstance(inp, str):
        try:
            info = OTBObject("ReadImageInfo", inp, quiet=True)
        except Exception as info_err:  # this happens when we pass a str that is not a filepath
            raise TypeError(f"Could not get the pixel type of `{inp}`. Not a filepath or wrong filepath") from info_err
        datatype = info.GetParameterString("datatype")  # which is such as short, float...
        if not datatype:
            raise TypeError(f"Unable to read pixel type of image {inp}")
        datatype_to_pixeltype = {
            'unsigned_char': 'uint8',
            'short': 'int16',
            'unsigned_short': 'uint16',
            'int': 'int32',
            'unsigned_int': 'uint32',
            'long': 'int32',
            'ulong': 'uint32',
            'float': 'float',
            'double': 'double'
        }
        if datatype not in datatype_to_pixeltype:
            raise TypeError(f"Unknown data type `{datatype}`. Available ones: {datatype_to_pixeltype}")
        pixel_type = getattr(otb, f'ImagePixelType_{datatype_to_pixeltype[datatype]}')
    elif isinstance(inp, OTBObject):
        pixel_type = inp.GetParameterOutputImagePixelType(inp.key_output_image)
    else:
        raise TypeError(f'Could not get the pixel type of {type(inp)} object {inp}')
    return pixel_type


def parse_pixel_type(pixel_type: str | int) -> int:
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


def is_key_list(pyotb_app: OTBObject, key: str) -> bool:
    """Check if a key of the App is an input parameter list."""
    return pyotb_app.app.GetParameterType(key) in (
        otb.ParameterType_InputImageList,
        otb.ParameterType_StringList,
        otb.ParameterType_InputFilenameList,
        otb.ParameterType_ListView,
        otb.ParameterType_InputVectorDataList,
    )


def is_key_images_list(pyotb_app: OTBObject, key: str) -> bool:
    """Check if a key of the App is an input parameter image list."""
    return pyotb_app.app.GetParameterType(key) in (
        otb.ParameterType_InputImageList,
        otb.ParameterType_InputFilenameList
    )


def get_out_images_param_keys(app: OTBObject) -> list[str]:
    """Return every output parameter keys of an OTB app."""
    return [key for key in app.GetParametersKeys() if app.GetParameterType(key) == otb.ParameterType_OutputImage]
