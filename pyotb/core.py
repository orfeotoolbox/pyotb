# -*- coding: utf-8 -*-
"""This module is the core of pyotb."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import otbApplication as otb  # pylint: disable=import-error

from .helpers import logger
from .depreciation import deprecated_alias, depreciation_warning, deprecated_attr


class OTBObject(ABC):
    """Abstraction of an image object."""

    @property
    @abstractmethod
    def name(self) -> str:
        """By default, should return the application name, but a custom name may be passed during init."""

    @property
    @abstractmethod
    def app(self) -> otb.Application:
        """Reference to the main (or last in pipeline) otb.Application instance linked to this object."""

    @property
    @abstractmethod
    def output_image_key(self) -> str:
        """Return the name of a parameter key associated to the main output image of the object."""

    @property
    @deprecated_attr(replacement="output_image_key")
    def output_param(self) -> str:
        """Return the name of a parameter key associated to the main output image of the object (deprecated)."""

    @property
    @abstractmethod
    def exports_dic(self) -> dict[str, dict]:
        """Return an internal dict object containing np.array exports, to avoid duplicated ExportImage() calls."""

    @property
    def metadata(self) -> dict[str, (str, float, list[float])]:
        """Return metadata.

        The returned dict results from the concatenation of the first output
        image metadata dictionary and the metadata dictionary.

        """
        # Image Metadata
        otb_imd = self.app.GetImageMetadata(self.output_image_key)
        cats = ["Num", "Str", "L1D", "Time"]
        imd = {
            key: getattr(otb_imd, f"get_{cat.lower()}")(key)
            for cat in cats
            for key in getattr(otb_imd, f"GetKeyList{cat}")().split(" ")
            if getattr(otb_imd, "has")(key)
        }

        # Metadata dictionary
        # Replace items like {"metadata_1": "TIFFTAG_SOFTWARE=CSinG - 13
        # SEPTEMBRE 2012"} with {"TIFFTAG_SOFTWARE": "CSinG - 13 SEPTEMBRE
        # 2012"}
        mdd = dict(self.app.GetMetadataDictionary(self.output_image_key))
        new_mdd = {}
        for key, val in mdd.items():
            new_key = key
            new_val = val
            if isinstance(val, str):
                splits = val.split("=")
                if key.lower().startswith("metadata_") and len(splits) == 2:
                    new_key = splits[0].strip()
                    new_val = splits[1].strip()
            new_mdd[new_key] = new_val

        return {**new_mdd, **imd}

    @property
    def dtype(self) -> np.dtype:
        """Expose the pixel type of output image using numpy convention.

        Returns:
            dtype: pixel type of the output image

        """
        enum = self.app.GetParameterOutputImagePixelType(self.output_image_key)
        return self.app.ConvertPixelTypeToNumpy(enum)

    @property
    def shape(self) -> tuple[int]:
        """Enables to retrieve the shape of a pyotb object using numpy convention.

        Returns:
            shape: (height, width, bands)

        """
        width, height = self.app.GetImageSize(self.output_image_key)
        bands = self.app.GetImageNbBands(self.output_image_key)
        return height, width, bands

    @property
    def transform(self) -> tuple[int]:
        """Get image affine transform, rasterio style (see https://www.perrygeo.com/python-affine-transforms.html).

        Returns:
            transform: (X spacing, X offset, X origin, Y offset, Y spacing, Y origin)
        """
        spacing_x, spacing_y = self.app.GetImageSpacing(self.output_image_key)
        origin_x, origin_y = self.app.GetImageOrigin(self.output_image_key)
        # Shift image origin since OTB is giving coordinates of pixel center instead of corners
        origin_x, origin_y = origin_x - spacing_x / 2, origin_y - spacing_y / 2
        return spacing_x, 0.0, origin_x, 0.0, spacing_y, origin_y

    def summarize(self, *args, **kwargs):
        """Recursively summarize parameters and parents.

        Args:
            *args: args for `pyotb.summarize()`
            **kwargs: keyword args for `pyotb.summarize()`

        Returns:
            app summary, same as `pyotb.summarize()`

        """
        return summarize(self, *args, **kwargs)

    def get_info(self) -> dict[str, (str, float, list[float])]:
        """Return a dict output of ReadImageInfo for the first image output."""
        return App("ReadImageInfo", self, quiet=True).data

    def get_statistics(self) -> dict[str, (str, float, list[float])]:
        """Return a dict output of ComputeImagesStatistics for the first image output."""
        return App("ComputeImagesStatistics", self, quiet=True).data

    def get_values_at_coords(
        self, row: int, col: int, bands: int = None
    ) -> list[int | float] | int | float:
        """Get pixel value(s) at a given YX coordinates.

        Args:
            row: index along Y / latitude axis
            col: index along X / longitude axis
            bands: band number, list or slice to fetch values from

        Returns:
            single numerical value or a list of values for each band

        """
        channels = []
        app = App("PixelValue", self, coordx=col, coordy=row, frozen=True, quiet=True)
        if bands is not None:
            if isinstance(bands, int):
                if bands < 0:
                    bands = self.shape[2] + bands
                channels = [bands]
            elif isinstance(bands, slice):
                channels = self.channels_list_from_slice(bands)
            elif not isinstance(bands, list):
                raise TypeError(
                    f"{self.name}: type '{type(bands)}' cannot be interpreted as a valid slicing"
                )
            if channels:
                app.app.Execute()
                app.set_parameters({"cl": [f"Channel{n + 1}" for n in channels]})
        app.execute()
        data = literal_eval(app.app.GetParameterString("value"))
        return data[0] if len(channels) == 1 else data

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
        raise ValueError(
            f"{self.name}: '{bands}' cannot be interpreted as valid slicing."
        )

    def export(
        self, key: str = None, preserve_dtype: bool = True
    ) -> dict[str, dict[str, np.ndarray]]:
        """Export a specific output image as numpy array and store it in object exports_dic.

        Args:
            key: parameter key to export, if None then the default one will be used
            preserve_dtype: when set to True, the numpy array is converted to the same pixel type as
                            the App first output. Default is True

        Returns:
            the exported numpy array

        """
        if key is None:
            key = self.output_image_key
        if key not in self.exports_dic:
            self.exports_dic[key] = self.app.ExportImage(key)
        if preserve_dtype:
            self.exports_dic[key]["array"] = self.exports_dic[key]["array"].astype(
                self.dtype
            )
        return self.exports_dic[key]

    def to_numpy(
        self, key: str = None, preserve_dtype: bool = True, copy: bool = False
    ) -> np.ndarray:
        """Export a pyotb object to numpy array.

        Args:
            key: the output parameter name to export as numpy array
            preserve_dtype: when set to True, the numpy array is converted to the same pixel type as
                            the App first output. Default is True
            copy: whether to copy the output array, default is False
                  required to True if preserve_dtype is False and the source app reference is lost

        Returns:
            a numpy array

        """
        data = self.export(key, preserve_dtype)
        return data["array"].copy() if copy else data["array"]

    def to_rasterio(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Export image as a numpy array and its metadata compatible with rasterio.

        Returns:
          array : a numpy array in the (bands, height, width) order
          profile: a metadata dict required to write image using rasterio

        """
        profile = {}
        array = self.to_numpy(preserve_dtype=True, copy=False)
        proj = self.app.GetImageProjection(self.output_image_key)
        profile.update({"crs": proj, "dtype": array.dtype, "transform": self.transform})
        height, width, count = array.shape
        profile.update({"count": count, "height": height, "width": width})
        return np.moveaxis(array, 2, 0), profile

    def get_rowcol_from_xy(self, x: float, y: float) -> tuple[int, int]:
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

    @staticmethod
    def __create_operator(op_cls, name, x, y) -> Operation:
        """Create an operator.

        Args:
            op_cls: Operator class
            name: operator expression
            x: first element
            y: second element

        Return:
            operator

        """
        if isinstance(y, (np.ndarray, np.generic)):
            return NotImplemented  # this enables to fallback on numpy emulation thanks to __array_ufunc__
        return op_cls(name, x, y)

    def __add__(self, other: OTBObject | str | int | float) -> Operation:
        """Addition."""
        return self.__create_operator(Operation, "+", self, other)

    def __sub__(self, other: OTBObject | str | int | float) -> Operation:
        """Subtraction."""
        return self.__create_operator(Operation, "-", self, other)

    def __mul__(self, other: OTBObject | str | int | float) -> Operation:
        """Multiplication."""
        return self.__create_operator(Operation, "*", self, other)

    def __truediv__(self, other: OTBObject | str | int | float) -> Operation:
        """Division."""
        return self.__create_operator(Operation, "/", self, other)

    def __radd__(self, other: OTBObject | str | int | float) -> Operation:
        """Right addition."""
        return self.__create_operator(Operation, "+", other, self)

    def __rsub__(self, other: OTBObject | str | int | float) -> Operation:
        """Right subtraction."""
        return self.__create_operator(Operation, "-", other, self)

    def __rmul__(self, other: OTBObject | str | int | float) -> Operation:
        """Right multiplication."""
        return self.__create_operator(Operation, "*", other, self)

    def __rtruediv__(self, other: OTBObject | str | int | float) -> Operation:
        """Right division."""
        return self.__create_operator(Operation, "/", other, self)

    def __abs__(self) -> Operation:
        """Absolute value."""
        return Operation("abs", self)

    def __ge__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Greater of equal than."""
        return self.__create_operator(LogicalOperation, ">=", self, other)

    def __le__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Lower of equal than."""
        return self.__create_operator(LogicalOperation, "<=", self, other)

    def __gt__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Greater than."""
        return self.__create_operator(LogicalOperation, ">", self, other)

    def __lt__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Lower than."""
        return self.__create_operator(LogicalOperation, "<", self, other)

    def __eq__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Equality."""
        return self.__create_operator(LogicalOperation, "==", self, other)

    def __ne__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Inequality."""
        return self.__create_operator(LogicalOperation, "!=", self, other)

    def __or__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Logical or."""
        return self.__create_operator(LogicalOperation, "||", self, other)

    def __and__(self, other: OTBObject | str | int | float) -> LogicalOperation:
        """Logical and."""
        return self.__create_operator(LogicalOperation, "&&", self, other)

    # Some other operations could be implemented with the same pattern
    # e.g. __pow__... cf https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __array__(self) -> np.ndarray:
        """This is called when running np.asarray(pyotb_object).

        Returns:
            a numpy array

        """
        return self.to_numpy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> App:
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
                    image_dic = inp.export()
                    arrays.append(image_dic["array"])
                else:
                    logger.debug(type(self))
                    return NotImplemented
            # Performing the numpy operation
            result_array = ufunc(*arrays, **kwargs)
            result_dic = image_dic
            result_dic["array"] = result_array
            # Importing back to OTB, pass the result_dic just to keep reference
            pyotb_app = App("ExtractROI", frozen=True, quiet=True)
            if result_array.shape[2] == 1:
                pyotb_app.app.ImportImage("in", result_dic)
            else:
                pyotb_app.app.ImportVectorImage("in", result_dic)
            pyotb_app.execute()
            return pyotb_app
        return NotImplemented

    def __hash__(self) -> int:
        """Override the default behaviour of the hash function.

        Returns:
            self hash

        """
        return id(self)

    def __getattr__(self, item: str):
        """Provides depreciation of old methods to access the OTB application values.

        This function will be removed completely in future releases.

        Args:
            item: attribute name


        """
        note = (
            "Since pyotb 2.0.0, OTBObject instances have stopped to forward "
            "attributes to their own internal otbApplication instance. "
            "`App.app` can be used to call otbApplications methods."
        )
        hint = None

        if item in dir(self.app):
            # Because otbApplication instances methods names start with an
            # upper case
            hint = f"Maybe try `pyotb_app.app.{item}` instead of `pyotb_app.{item}`? "
            if item.startswith("GetParameter"):
                hint += (
                    "Note: `pyotb_app.app.GetParameterValue('paramname')` can be "
                    "shorten with `pyotb_app['paramname']` to access parameters "
                    "values."
                )

        elif item in self.parameters_keys:
            # Because in pyotb 1.5.4, applications outputs were added as
            # attributes of the instance
            hint = (
                "Note: `pyotb_app.paramname` is no longer supported. Starting "
                "from pyotb 2.0.0, `pyotb_app['paramname']` can be used to "
                "access parameters values. "
            )
        if hint:
            depreciation_warning(f"{note} {hint}")
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{item}'"
        )

    def __getitem__(self, key) -> Any | list[float] | float | Slicer:
        """Override the default __getitem__ behaviour.

        This function enables 2 things :
        - slicing, i.e. selecting ROI/bands. For example, selecting first 3 bands: object[:, :, :3]
                                                          selecting bands 1, 2 & 5 : object[:, :, [0, 1, 4]]
                                                          selecting 1000x1000 subset : object[:1000, :1000]
        - access pixel value(s) at a specified row, col index

        Args:
            key: attribute key

        Returns:
            list of pixel values if vector image, or pixel value, or Slicer

        """
        # Accessing pixel value(s) using Y/X coordinates
        if isinstance(key, tuple) and len(key) >= 2:
            row, col = key[0], key[1]
            if isinstance(row, int) and isinstance(col, int):
                if row < 0 or col < 0:
                    raise ValueError(
                        f"{self.name} cannot read pixel value at negative coordinates ({row}, {col})"
                    )
                channels = key[2] if len(key) == 3 else None
                return self.get_values_at_coords(row, col, channels)
        # Slicing
        if not isinstance(key, tuple) or (
            isinstance(key, tuple) and (len(key) < 2 or len(key) > 3)
        ):
            raise ValueError(
                f'"{key}" cannot be interpreted as valid slicing. Slicing should be 2D or 3D.'
            )
        if isinstance(key, tuple) and len(key) == 2:
            # Adding a 3rd dimension
            key = key + (slice(None, None, None),)
        return Slicer(self, *key)

    def __repr__(self) -> str:
        """Return a string representation with object id, this is a key used to store image ref in Operation dicts."""
        return f"<pyotb.{self.__class__.__name__} object, id {id(self)}>"


class App(OTBObject):
    """Base class that gathers common operations for any OTB application."""

    INPUT_IMAGE_TYPES = [
        otb.ParameterType_InputImage,
        otb.ParameterType_InputImageList,
    ]
    INPUT_PARAM_TYPES = INPUT_IMAGE_TYPES + [
        otb.ParameterType_InputVectorData,
        otb.ParameterType_InputVectorDataList,
        otb.ParameterType_InputFilename,
        otb.ParameterType_InputFilenameList,
    ]

    OUTPUT_IMAGE_TYPES = [otb.ParameterType_OutputImage]
    OUTPUT_PARAM_TYPES = OUTPUT_IMAGE_TYPES + [
        otb.ParameterType_OutputVectorData,
        otb.ParameterType_OutputFilename,
    ]

    INPUT_LIST_TYPES = [
        otb.ParameterType_InputImageList,
        otb.ParameterType_StringList,
        otb.ParameterType_InputFilenameList,
        otb.ParameterType_ListView,
        otb.ParameterType_InputVectorDataList,
    ]
    INPUT_IMAGES_LIST_TYPES = [
        otb.ParameterType_InputImageList,
        otb.ParameterType_InputFilenameList,
    ]

    def __init__(
        self,
        appname: str,
        *args,
        frozen: bool = False,
        quiet: bool = False,
        name: str = "",
        **kwargs,
    ):
        """Common constructor for OTB applications. Handles in-memory connection between apps.

        Args:
            appname: name of the OTB application to initialize, e.g. 'BandMath'
            *args: used for passing application parameters. Can be :
                           - dictionary containing key-arguments enumeration. Useful when a key is python-reserved
                             (e.g. "in") or contains reserved characters such as a point (e.g."mode.extent.unit")
                           - string, App or Output, useful when the user wants to specify the input "in"
                           - list, useful when the user wants to specify the input list 'il'
            frozen: freeze OTB app in order to use execute() later and avoid blocking process during __init___
            quiet: whether to print logs of the OTB app
            name: custom name that will show up in logs, appname will be used if not provided

            **kwargs: used for passing application parameters.
                      e.g. il=['input1.tif', App_object2, App_object3.out], out='output.tif'

        """
        # Attributes and data structures used by properties
        create = (
            otb.Registry.CreateApplicationWithoutLogger
            if quiet
            else otb.Registry.CreateApplication
        )
        self._app = create(appname)
        self._name = name or appname
        self._exports_dic = {}
        self._settings, self._auto_parameters = {}, {}
        self._time_start, self._time_end = 0.0, 0.0
        self.data, self.outputs = {}, {}
        self.quiet, self.frozen = quiet, frozen
        # Param keys and types
        self.parameters_keys = tuple(self.app.GetParametersKeys())
        self._all_param_types = {
            key: self.app.GetParameterType(key) for key in self.parameters_keys
        }
        self._out_param_types = {
            key: val
            for key, val in self._all_param_types.items()
            if val in self.OUTPUT_PARAM_TYPES
        }
        self._key_choices = {
            key: [f"{key}.{choice}" for choice in self.app.GetChoiceKeys(key)]
            for key in self.parameters_keys
            if self.app.GetParameterType(key) == otb.ParameterType_Choice
        }
        # Init, execute and write (auto flush only when output param was provided)
        if args or kwargs:
            self.set_parameters(*args, **kwargs)
        # Create Output image objects
        for key in filter(
            lambda k: self._out_param_types[k] == otb.ParameterType_OutputImage,
            self._out_param_types,
        ):
            self.outputs[key] = Output(self, key, self._settings.get(key))
        if not self.frozen:
            self.execute()
            if any(key in self._settings for key in self._out_param_types):
                self.flush()
        else:
            self.__sync_parameters()  # since not called during execute()

    @property
    def name(self) -> str:
        """Returns appname by default, or a custom name if passed during App init."""
        return self._name

    @property
    def app(self) -> otb.Application:
        """Property to return an internal _app instance."""
        return self._app

    @property
    def parameters(self):
        """Return used application parameters: automatic values or set by user."""
        return {**self._auto_parameters, **self._settings}

    @property
    def exports_dic(self) -> dict[str, dict]:
        """Returns internal _exports_dic object that contains numpy array exports."""
        return self._exports_dic

    def __is_one_of_types(self, key: str, param_types: list[int]) -> bool:
        """Helper to factor is_input and is_output."""
        if key not in self._all_param_types:
            raise KeyError(f"key {key} not found in the application parameters types")
        return self._all_param_types[key] in param_types

    def __is_multi_output(self):
        """Check if app has multiple output since it needs exec during write()."""
        return len(self.outputs) > 1

    def is_input(self, key: str) -> bool:
        """Returns True if the key is an input.

        Args:
            key: parameter key

        Returns:
            True if the parameter is an input, else False

        """
        return self.__is_one_of_types(key=key, param_types=self.INPUT_PARAM_TYPES)

    def is_output(self, key: str) -> bool:
        """Returns True if the key is an output.

        Args:
            key: parameter key

        Returns:
            True if the parameter is an output, else False

        """
        return self.__is_one_of_types(key=key, param_types=self.OUTPUT_PARAM_TYPES)

    def is_key_list(self, key: str) -> bool:
        """Check if a parameter key is an input parameter list."""
        return self.app.GetParameterType(key) in self.INPUT_LIST_TYPES

    def is_key_images_list(self, key: str) -> bool:
        """Check if a parameter key is an input parameter image list."""
        return self.app.GetParameterType(key) in self.INPUT_IMAGES_LIST_TYPES

    def get_first_key(self, param_types: list[int]) -> str:
        """Get the first param key for specific file types, try each list in args."""
        for param_type in param_types:
            # Return the first key, from the alphabetically sorted keys of the
            # application, which has the parameter type matching param_type.
            for key, value in sorted(self._all_param_types.items()):
                if value == param_type:
                    return key
        raise TypeError(
            f"{self.name}: could not find any parameter key matching the provided types"
        )

    @property
    def input_key(self) -> str:
        """Get the name of first input parameter, raster > vector > file."""
        return self.get_first_key(self.INPUT_PARAM_TYPES)

    @property
    def input_image_key(self) -> str:
        """Name of the first input image parameter."""
        return self.get_first_key(self.INPUT_IMAGE_TYPES)

    @property
    def output_key(self) -> str:
        """Name of the first output parameter, raster > vector > file."""
        return self.get_first_key(self.OUTPUT_PARAM_TYPES)

    @property
    def output_image_key(self) -> str:
        """Get the name of first output image parameter."""
        return self.get_first_key(self.OUTPUT_IMAGE_TYPES)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time between app init and end of exec or file writing."""
        return self._time_end - self._time_start

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
            if "_" in key:
                key = key.replace("_", ".")
            if key not in self.parameters_keys:
                raise KeyError(
                    f"{self.name}: parameter '{key}' was not recognized. Available keys are {self.parameters_keys}"
                )
            # When the parameter expects a list, if needed, change the value to list
            if self.is_key_list(key) and not isinstance(obj, (list, tuple)):
                obj = [obj]
                logger.info(
                    '%s: argument for parameter "%s" was converted to list',
                    self.name,
                    key,
                )
            try:
                if self.is_input(key):
                    obj = self.__check_input_param(obj)
                elif self.is_output(key):
                    obj = self.__check_output_param(obj)
                self.__set_param(key, obj)
            except (RuntimeError, TypeError, ValueError, KeyError) as e:
                raise RuntimeError(
                    f"{self.name}: error before execution, while setting parameter '{key}' to '{obj}': {e})"
                ) from e
            # Save / update setting value and update the Output object initialized in __init__ without a filepath
            self._settings[key] = obj
            if key in self.outputs:
                self.outputs[key].filepath = obj
            if key in self._auto_parameters:
                del self._auto_parameters[key]

    def propagate_dtype(self, target_key: str = None, dtype: int = None):
        """Propagate a pixel type from main input to every outputs, or to a target output key only.

        With multiple inputs (if dtype is not provided), the type of the first input is considered.
        With multiple outputs (if target_key is not provided), all outputs will be converted to the same pixel type.

        Args:
            target_key: output param key to change pixel type
            dtype: data type to use

        """
        if not dtype:
            param = self._settings.get(self.input_image_key)
            if not param:
                logger.warning(
                    "%s: could not propagate pixel type from inputs to output",
                    self.name,
                )
                return
            if isinstance(param, (list, tuple)):
                param = param[0]  # first image in "il"
            try:
                dtype = get_pixel_type(param)
            except (TypeError, RuntimeError):
                logger.warning(
                    '%s: unable to identify pixel type of key "%s"', self.name, param
                )
                return
        if target_key:
            keys = [target_key]
        else:
            keys = [
                k
                for k, v in self._out_param_types.items()
                if v == otb.ParameterType_OutputImage
            ]
        for key in keys:
            self.app.SetParameterOutputImagePixelType(key, dtype)

    def execute(self):
        """Execute and write to disk if any output parameter has been set during init."""
        logger.debug("%s: run execute() with parameters=%s", self.name, self.parameters)
        self._time_start = perf_counter()
        try:
            self.app.Execute()
        except (RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(
                f"{self.name}: error during during app execution ({e}"
            ) from e
        self.frozen = False
        self._time_end = perf_counter()
        logger.debug("%s: execution ended", self.name)
        self.__sync_parameters()

    def flush(self):
        """Flush data to disk, this is when WriteOutput is actually called."""
        logger.debug("%s: flushing data to disk", self.name)
        self.app.WriteOutput()
        self._time_end = perf_counter()

    @deprecated_alias(filename_extension="ext_fname")
    def write(
        self,
        path: str | Path | dict[str, str] = None,
        pixel_type: dict[str, str] | str = None,
        preserve_dtype: bool = False,
        ext_fname: dict[str, str] | str = None,
        **kwargs,
    ) -> bool:
        """Set output pixel type and write the output raster files.

        Args:
            path: Can be : - filepath, useful when there is only one output, e.g. 'output.tif'
                           - dictionary containing key-arguments enumeration. Useful when a key contains
                             non-standard characters such as a point, e.g. {'io.out':'output.tif'}
                           - None if output file was passed during App init
            pixel_type: Can be : - dictionary {out_param_key: pixeltype} when specifying for several outputs
                                 - str (e.g. 'uint16') or otbApplication.ImagePixelType_... When there are several
                                   outputs, all outputs are written with this unique type.
                                   Valid pixel types are uint8, uint16, uint32, int16, int32, float, double,
                                   cint16, cint32, cfloat, cdouble. (Default value = None)
            preserve_dtype: propagate main input pixel type to outputs, in case pixel_type is None
            ext_fname: Optional, an extended filename as understood by OTB (e.g. "&gdal:co:TILED=YES")
                                Will be used for all outputs (Default value = None)
            **kwargs: keyword arguments e.g. out='output.tif'

        Returns:
            True if all files are found on disk

        """
        # Gather all input arguments in kwargs dict
        if isinstance(path, dict):
            kwargs.update(path)
        elif isinstance(path, str) and kwargs:
            logger.warning(
                '%s: keyword arguments specified, ignoring argument "%s"',
                self.name,
                path,
            )
        elif isinstance(path, (str, Path)) and self.output_key:
            kwargs[self.output_key] = str(path)
        elif not path and self.output_key in self.parameters:
            kwargs[self.output_key] = self.parameters[self.output_key]
        elif path is not None:
            raise TypeError(f"{self.name}: unsupported filepath type ({type(path)})")
        if not (kwargs or any(k in self._settings for k in self._out_param_types)):
            raise KeyError(
                f"{self.name}: at least one filepath is required, if not provided during App init"
            )
        parameters = kwargs.copy()

        # Append filename extension to filenames
        if ext_fname:
            if not isinstance(ext_fname, (dict, str)):
                raise ValueError("Extended filename must be a str or a dict")

            def _str2dict(ext_str):
                """Function that converts str to dict."""
                splits = [pair.split("=") for pair in ext_str.split("&")]
                return dict(split for split in splits if len(split) == 2)

            if isinstance(ext_fname, str):
                ext_fname = _str2dict(ext_fname)

            logger.debug("%s: extended filename for all outputs:", self.name)
            for key, ext in ext_fname.items():
                logger.debug("%s: %s", key, ext)

            for key, filepath in kwargs.items():
                if self._out_param_types[key] == otb.ParameterType_OutputImage:
                    new_ext_fname = ext_fname.copy()

                    # grab already set extended filename key/values
                    if "?&" in filepath:
                        filepath, already_set_ext = filepath.split("?&", 1)
                        # extensions in filepath prevail over `new_ext_fname`
                        new_ext_fname.update(_str2dict(already_set_ext))

                    # transform dict to str
                    ext_fname_str = "&".join(
                        [f"{key}={value}" for key, value in new_ext_fname.items()]
                    )
                    parameters[key] = f"{filepath}?&{ext_fname_str}"

        # Manage output pixel types
        data_types = {}
        if pixel_type:
            if isinstance(pixel_type, str):
                dtype = parse_pixel_type(pixel_type)
                type_name = self.app.ConvertPixelTypeToNumpy(dtype)
                logger.debug(
                    '%s: output(s) will be written with type "%s"', self.name, type_name
                )
                for key in parameters:
                    if self._out_param_types[key] == otb.ParameterType_OutputImage:
                        data_types[key] = dtype
            elif isinstance(pixel_type, dict):
                data_types = {
                    key: parse_pixel_type(dtype) for key, dtype in pixel_type.items()
                }
        elif preserve_dtype:
            self.propagate_dtype()  # all outputs will have the same type as the main input raster

        # Set parameters and flush to disk
        for key, filepath in parameters.items():
            if Path(filepath.split("?")[0]).exists():
                logger.warning("%s: overwriting file %s", self.name, filepath)
            if key in data_types:
                self.propagate_dtype(key, data_types[key])
            self.set_parameters({key: filepath})
        if self.frozen or self.__is_multi_output():
            self.execute()
        self.flush()
        if not parameters:
            return True

        # Search and log missing files
        files, missing = [], []
        for key, filepath in parameters.items():
            if not filepath.startswith("/vsi"):
                filepath = Path(filepath.split("?")[0])
                dest = files if filepath.exists() else missing
                dest.append(str(filepath.absolute()))
        for filename in missing:
            logger.error(
                "%s: execution seems to have failed, %s does not exist",
                self.name,
                filename,
            )
        return bool(files) and not missing

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
            elif (
                isinstance(arg, (str, OTBObject))
                or isinstance(arg, list)
                and self.is_key_list(self.input_key)
            ):
                kwargs.update({self.input_key: arg})
        return kwargs

    def __check_input_param(
        self, obj: list | OTBObject | str | Path
    ) -> list | OTBObject | str:
        """Check the type and value of an input parameter, add vsi prefixes if needed."""
        if isinstance(obj, list):
            return [self.__check_input_param(o) for o in obj]
        # May be we could add some checks here
        if isinstance(obj, OTBObject):
            return obj
        if isinstance(obj, Path):
            obj = str(obj)
        if isinstance(obj, str):
            if not obj.startswith("/vsi"):
                # Remote file. TODO: add support for S3 / GS / AZ
                if obj.startswith(("https://", "http://", "ftp://")):
                    obj = "/vsicurl/" + obj
                # Compressed file
                prefixes = {
                    ".tar": "vsitar",
                    ".tar.gz": "vsitar",
                    ".tgz": "vsitar",
                    ".gz": "vsigzip",
                    ".7z": "vsi7z",
                    ".zip": "vsizip",
                    ".rar": "vsirar",
                }
                expr = r"(.*?)(\.7z|\.zip|\.rar|\.tar\.gz|\.tgz|\.tar|\.gz)(.*)"
                parts = re.match(expr, obj)
                if parts:
                    file, ext = parts.group(1), parts.group(2)
                    if not Path(file + ext).is_dir():
                        obj = f"/{prefixes[ext]}/{obj}"
            return obj
        raise TypeError(f"{self.name}: wrong input parameter type ({type(obj)})")

    def __check_output_param(self, obj: list | str | Path) -> list | str:
        """Check the type and value of an output parameter."""
        if isinstance(obj, list):
            return [self.__check_output_param(o) for o in obj]
        if isinstance(obj, Path):
            obj = str(obj)
        if isinstance(obj, str):
            return obj
        raise TypeError(f"{self.name}: wrong output parameter type ({type(obj)})")

    def __set_param(
        self, key: str, obj: str | float | list | tuple | OTBObject | otb.Application
    ):
        """Set one parameter, decide which otb.Application method to use depending on target object."""
        if obj is None or (isinstance(obj, (list, tuple)) and not obj):
            self.app.ClearValue(key)
            return
        # Single-parameter cases
        if isinstance(obj, OTBObject):
            self.app.ConnectImage(key, obj.app, obj.output_image_key)
        elif isinstance(
            obj, otb.Application
        ):  # this is for backward comp with plain OTB
            self.app.ConnectImage(key, obj, get_out_images_param_keys(obj)[0])
        elif (
            key == "ram"
        ):  # SetParameterValue in OTB<7.4 doesn't work for ram parameter cf gitlab OTB issue 2200
            self.app.SetParameterInt("ram", int(obj))
        elif not isinstance(obj, list):  # any other parameters (str, int...)
            self.app.SetParameterValue(key, obj)
        # Images list
        elif self.is_key_images_list(key):
            # To enable possible in-memory connections, we go through the list and set the parameters one by one
            for inp in obj:
                if isinstance(inp, OTBObject):
                    self.app.ConnectImage(key, inp.app, inp.output_image_key)
                elif isinstance(
                    inp, otb.Application
                ):  # this is for backward comp with plain OTB
                    self.app.ConnectImage(key, obj, get_out_images_param_keys(inp)[0])
                else:  # here `input` should be an image filepath
                    # Append `input` to the list, do not overwrite any previously set element of the image list
                    self.app.AddParameterStringList(key, inp)
        # List of any other types (str, int...)
        else:
            self.app.SetParameterValue(key, obj)

    def __sync_parameters(self):
        """Save app parameters in _auto_parameters or data dict.

        This is always called during init or after execution, to ensure the
         parameters property of the App is in sync with the otb.Application instance.
        """
        skip = [
            k for k in self.parameters_keys if k.split(".")[-1] in ("ram", "default")
        ]
        # Prune unused choices child params
        for key in self._key_choices:
            choices = self._key_choices[key].copy()
            choices.remove(f"{key}.{self.app.GetParameterValue(key)}")
            skip.extend(
                [k for k in self.parameters_keys if k.startswith(tuple(choices))]
            )

        self._auto_parameters.clear()
        for key in self.parameters_keys:
            if key in skip or key in self._settings or not self.app.HasValue(key):
                continue
            value = self.app.GetParameterValue(key)
            if isinstance(value, otb.ApplicationProxy):
                try:
                    value = str(value)
                except RuntimeError:
                    continue
            # Keep False or 0 values, but make sure to skip empty collections or str
            if hasattr(value, "__iter__") and not value:
                continue
            # Here we should use AND self.app.IsParameterEnabled(key) but it's broken
            if self.app.GetParameterRole(key) == 0 and (
                self.app.HasAutomaticValue(key) or self.app.IsParameterEnabled(key)
            ):
                self._auto_parameters[key] = value
            # Save static output data (ReadImageInfo, ComputeImageStatistics, etc.)
            elif self.app.GetParameterRole(key) == 1:
                if isinstance(value, str):
                    try:
                        value = literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
                self.data[key] = value

    # Special functions
    def __getitem__(self, key: str) -> Any | list[int | float] | int | float | Slicer:
        """This function is called when we use App()[...].

        We allow to return attr if key is a parameter, or call OTBObject __getitem__ for pixel values or Slicer
        """
        if isinstance(key, tuple):
            return super().__getitem__(key)  # to read pixel values, or slice
        if isinstance(key, str):
            if key in self.data:
                return self.data[key]
            if key in self.outputs:
                return self.outputs[key]
            if key in self.parameters:
                return self.parameters[key]
            raise KeyError(f"{self.name}: unknown or undefined parameter '{key}'")
        raise TypeError(
            f"{self.name}: cannot access object item or slice using {type(key)} object"
        )


class Slicer(App):
    """Slicer objects i.e. when we call something like raster[:, :, 2] from Python."""

    def __init__(
        self,
        obj: OTBObject,
        rows: slice,
        cols: slice,
        channels: slice | list[int] | int,
    ):
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
        super().__init__(
            "ExtractROI",
            obj,
            mode="extent",
            quiet=True,
            frozen=True,
            name=f"Slicer from {obj.name}",
        )
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
                raise ValueError(
                    f"Invalid type for channels, should be int, slice or list of bands. : {channels}"
                )
            # Change the potential negative index values to reverse index
            channels = [c if c >= 0 else nb_channels + c for c in channels]
            parameters.update({"cl": [f"Channel{i + 1}" for i in channels]})

        # Spatial slicing
        spatial_slicing = False
        # TODO: handle the step value in the slice so that NN undersampling is possible ? e.g. raster[::2, ::2]
        if rows.start is not None:
            parameters.update({"mode.extent.uly": rows.start})
            spatial_slicing = True
        if rows.stop is not None and rows.stop != -1:
            parameters.update(
                {"mode.extent.lry": rows.stop - 1}
            )  # subtract 1 to respect python convention
            spatial_slicing = True
        if cols.start is not None:
            parameters.update({"mode.extent.ulx": cols.start})
            spatial_slicing = True
        if cols.stop is not None and cols.stop != -1:
            parameters.update(
                {"mode.extent.lrx": cols.stop - 1}
            )  # subtract 1 to respect python convention
            spatial_slicing = True
        # These are some attributes when the user simply wants to extract *one* band to be used in an Operation
        if not spatial_slicing and isinstance(channels, list) and len(channels) == 1:
            self.one_band_sliced = (
                channels[0] + 1
            )  # OTB convention: channels start at 1
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

    def __init__(self, operator: str, *inputs, nb_bands: int = None, name: str = None):
        """Given some inputs and an operator, this function enables to transform this into an OTB application.

        Operations generally involve 2 inputs (+, -...). It can have only 1 input for `abs` operator.
        It can have 3 inputs for the ternary operator `cond ? x : y`.

        Args:
            operator: (str) one of +, -, *, /, >, <, >=, <=, ==, !=, &, |, abs, ?
            *inputs: inputs. Can be OTBObject, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where
            name: override the Operation name

        """
        self.operator = operator
        # We first create a 'fake' expression. E.g for the operation `input1 + input2` , we create a fake expression
        # that is like "str(input1) + str(input2)"
        self.inputs = []
        self.nb_channels = {}
        self.fake_exp_bands = []
        self.build_fake_expressions(operator, inputs, nb_bands=nb_bands)
        # Transforming images to the adequate im#, e.g. `input1` to "im1"
        # creating a dictionary that is like {str(input1): 'im1', 'image2.tif': 'im2', ...}.
        # NB: the keys of the dictionary are strings-only, instead of 'complex' objects, to enable easy serialization
        self.im_dic = {}
        self.im_count = 1
        map_repr_to_input = (
            {}
        )  # to be able to retrieve the real python object from its string representation
        for inp in self.inputs:
            if not isinstance(inp, (int, float)):
                if str(inp) not in self.im_dic:
                    self.im_dic[repr(inp)] = f"im{self.im_count}"
                    map_repr_to_input[repr(inp)] = inp
                    self.im_count += 1
        # Getting unique image inputs, in the order im1, im2, im3 ...
        self.unique_inputs = [
            map_repr_to_input[id_str]
            for id_str in sorted(self.im_dic, key=self.im_dic.get)
        ]
        self.exp_bands, self.exp = self.get_real_exp(self.fake_exp_bands)
        appname = "BandMath" if len(self.exp_bands) == 1 else "BandMathX"
        name = f'Operation exp="{self.exp}"'
        super().__init__(
            appname, il=self.unique_inputs, exp=self.exp, quiet=True, name=name
        )

    def build_fake_expressions(
        self,
        operator: str,
        inputs: list[OTBObject | str | int | float],
        nb_bands: int = None,
    ):
        """Create a list of 'fake' expressions, one for each band.

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
            if any(
                isinstance(inp, Slicer) and hasattr(inp, "one_band_sliced")
                for inp in inputs
            ):
                nb_bands = 1
            else:
                nb_bands_list = [
                    get_nbchannels(inp)
                    for inp in inputs
                    if not isinstance(inp, (float, int))
                ]
                # check that all inputs have the same nb of bands
                if len(nb_bands_list) > 1 and not all(
                    x == nb_bands_list[0] for x in nb_bands_list
                ):
                    raise ValueError("All images do not have the same number of bands")
                nb_bands = nb_bands_list[0]

        # Create a list of fake expressions, each item of the list corresponding to one band
        self.fake_exp_bands.clear()
        for i, band in enumerate(range(1, nb_bands + 1)):
            expressions = []
            for k, inp in enumerate(inputs):
                # Generating the fake expression of the current input,
                # this is a special case for the condition of the ternary operator `cond ? x : y`
                if len(inputs) == 3 and k == 0:
                    # When cond is monoband whereas the result is multiband, we expand the cond to multiband
                    cond_band = 1 if nb_bands != inp.shape[2] else band
                    fake_exp, corresponding_inputs, nb_channels = self.make_fake_exp(
                        inp, cond_band, keep_logical=True
                    )
                else:
                    # Any other input
                    fake_exp, corresponding_inputs, nb_channels = self.make_fake_exp(
                        inp, band, keep_logical=False
                    )
                expressions.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresponding_inputs and nb_channels:
                    self.inputs.extend(corresponding_inputs)
                    self.nb_channels.update(nb_channels)

            # Generating the fake expression of the whole operation
            if len(inputs) == 1:  # this is only for 'abs'
                fake_exp = f"({operator}({expressions[0]}))"
            elif len(inputs) == 2:
                # We create here the "fake" expression. For example, for a BandMathX expression such as '2 * im1 + im2',
                # the false expression stores the expression 2 * str(input1) + str(input2)
                fake_exp = f"({expressions[0]} {operator} {expressions[1]})"
            elif (
                len(inputs) == 3 and operator == "?"
            ):  # this is only for ternary expression
                fake_exp = f"({expressions[0]} ? {expressions[1]} : {expressions[2]})"
            self.fake_exp_bands.append(fake_exp)

    def get_real_exp(self, fake_exp_bands: str) -> tuple[list[str], str]:
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
                one_band_exp = one_band_exp.replace(repr(inp), self.im_dic[repr(inp)])
            exp_bands.append(one_band_exp)
        # Form the final expression (e.g. 'im1b1 + 1; im1b2 + 1')
        return exp_bands, ";".join(exp_bands)

    @staticmethod
    def make_fake_exp(
        x: OTBObject | str, band: int, keep_logical: bool = False
    ) -> tuple[str, list[OTBObject], int]:
        """This an internal function, only to be used by `build_fake_expressions`.

        Enable to create a fake expression just for one input and one band.

        Args:
            x: input
            band: which band to consider (bands start at 1)
            keep_logical: whether to keep the logical expressions "as is" in case the input is a logical operation.
                          ex: if True, for `input1 > input2`, returned fake expression is "str(input1) > str(input2)"
                          if False, for `input1 > input2`, returned fake exp is "str(input1) > str(input2) ? 1 : 0"]
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
                inputs, nb_channels = x.input.inputs, x.input.nb_channels
            elif isinstance(x.input, Operation):
                # Keep only one band of the expression
                fake_exp = x.input.fake_exp_bands[x.one_band_sliced - 1]
                inputs, nb_channels = x.input.inputs, x.input.nb_channels
            else:
                # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
                fake_exp = f"{repr(x.input)}b{x.one_band_sliced}"
                inputs, nb_channels = [x.input], {repr(x.input): 1}
        # For LogicalOperation, we save almost the same attributes as an Operation
        elif keep_logical and isinstance(x, LogicalOperation):
            fake_exp = x.logical_fake_exp_bands[band - 1]
            inputs, nb_channels = x.inputs, x.nb_channels
        elif isinstance(x, Operation):
            fake_exp = x.fake_exp_bands[band - 1]
            inputs, nb_channels = x.inputs, x.nb_channels
        # For int or float input, we just need to save their value
        elif isinstance(x, (int, float)):
            fake_exp = str(x)
            inputs, nb_channels = None, None
        # We go on with other inputs, i.e. pyotb objects, filepaths...
        else:
            # Add the band number (e.g. replace '<pyotb.App object>' by '<pyotb.App object>b1')
            fake_exp = f"{repr(x)}b{band}"
            inputs, nb_channels = [x], {repr(x): get_nbchannels(x)}

        return fake_exp, inputs, nb_channels

    def __repr__(self) -> str:
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
        self.logical_fake_exp_bands = []
        super().__init__(operator, *inputs, nb_bands=nb_bands, name="LogicalOperation")
        self.logical_exp_bands, self.logical_exp = self.get_real_exp(
            self.logical_fake_exp_bands
        )

    def build_fake_expressions(
        self,
        operator: str,
        inputs: list[OTBObject | str | int | float],
        nb_bands: int = None,
    ):
        """Create a list of 'fake' expressions, one for each band.

        e.g for the operation input1 > input2, we create a fake expression that is like
        "str(input1) > str(input2) ? 1 : 0" and a logical fake expression that is like "str(input1) > str(input2)"

        Args:
            operator: str (one of >, <, >=, <=, ==, !=, &, |)
            inputs: Can be OTBObject, filepath, int or float
            nb_bands: to specify the output nb of bands. Optional. Used only internally by pyotb.where

        """
        # For any other operations, the output number of bands is the same as inputs
        if any(
            isinstance(inp, Slicer) and hasattr(inp, "one_band_sliced")
            for inp in inputs
        ):
            nb_bands = 1
        else:
            nb_bands_list = [
                get_nbchannels(inp)
                for inp in inputs
                if not isinstance(inp, (float, int))
            ]
            # check that all inputs have the same nb of bands
            if len(nb_bands_list) > 1 and not all(
                x == nb_bands_list[0] for x in nb_bands_list
            ):
                raise ValueError("All images do not have the same number of bands")
            nb_bands = nb_bands_list[0]
        # Create a list of fake exp, each item of the list corresponding to one band
        for i, band in enumerate(range(1, nb_bands + 1)):
            expressions = []
            for inp in inputs:
                fake_exp, corresp_inputs, nb_channels = super().make_fake_exp(
                    inp, band, keep_logical=True
                )
                expressions.append(fake_exp)
                # Reference the inputs and nb of channels (only on first pass in the loop to avoid duplicates)
                if i == 0 and corresp_inputs and nb_channels:
                    self.inputs.extend(corresp_inputs)
                    self.nb_channels.update(nb_channels)
            # We create here the "fake" expression. For example, for a BandMathX expression such as 'im1 > im2',
            # the logical fake expression stores the expression "str(input1) > str(input2)"
            logical_fake_exp = f"({expressions[0]} {operator} {expressions[1]})"
            # We keep the logical expression, useful if later combined with other logical operations
            self.logical_fake_exp_bands.append(logical_fake_exp)
            # We create a valid BandMath expression, e.g. "str(input1) > str(input2) ? 1 : 0"
            fake_exp = f"({logical_fake_exp} ? 1 : 0)"
            self.fake_exp_bands.append(fake_exp)


class Input(App):
    """Class for transforming a filepath to pyOTB object."""

    def __init__(self, filepath: str):
        """Default constructor.

        Args:
            filepath: Anything supported by GDAL (local file on the filesystem, remote resource e.g. /vsicurl/.., etc.)

        """
        super().__init__("ExtractROI", {"in": filepath}, quiet=True, frozen=True)
        self._name = f"Input from {filepath}"
        if not filepath.startswith(("/vsi", "http://", "https://", "ftp://")):
            filepath = Path(filepath)
        self.filepath = filepath
        self.propagate_dtype()
        self.execute()

    def __repr__(self) -> str:
        """Return a string representation with file path, used in Operation to store file ref."""
        return f"<pyotb.Input object, from {self.filepath}>"


class Output(OTBObject):
    """Object that behave like a pointer to a specific application output file."""

    _filepath: str | Path = None

    @deprecated_alias(app="pyotb_app", output_parameter_key="param_key")
    def __init__(
        self,
        pyotb_app: App,
        param_key: str = None,
        filepath: str = None,
        mkdir: bool = True,
    ):
        """Constructor for an Output object.

        Args:
            pyotb_app: The pyotb App to store reference from
            param_key: Output parameter key of the target app
            filepath: path of the output file (if not in memory)
            mkdir: create missing parent directories

        """
        self.parent_pyotb_app = pyotb_app  # keep trace of parent app
        self.param_key = param_key
        self.filepath = filepath
        if mkdir and filepath is not None:
            self.make_parent_dirs()

    @property
    def name(self) -> str:
        """Return Output name containing filepath."""
        return f"Output {self.param_key} from {self.parent_pyotb_app.name}"

    @property
    def app(self) -> otb.Application:
        """Reference to the parent pyotb otb.Application instance."""
        return self.parent_pyotb_app.app

    @property
    @deprecated_attr(replacement="parent_pyotb_app")
    def pyotb_app(self) -> App:
        """Reference to the parent pyotb App (deprecated)."""

    @property
    def exports_dic(self) -> dict[str, dict]:
        """Returns internal _exports_dic object that contains numpy array exports."""
        return self.parent_pyotb_app.exports_dic

    @property
    def output_image_key(self) -> str:
        """Force the right key to be used when accessing the OTBObject."""
        return self.param_key

    @property
    def filepath(self) -> str | Path:
        """Property to manage output URL."""
        if self._filepath is None:
            raise ValueError("Filepath is not set")
        return self._filepath

    @filepath.setter
    def filepath(self, path: str):
        if isinstance(path, str):
            if path and not path.startswith(("/vsi", "http://", "https://", "ftp://")):
                path = Path(path.split("?")[0])
            self._filepath = path

    def exists(self) -> bool:
        """Check file exist."""
        if not isinstance(self.filepath, Path):
            raise ValueError("Filepath is not set or points to a remote URL")
        return self.filepath.exists()

    def make_parent_dirs(self):
        """Create missing parent directories."""
        if not isinstance(self.filepath, Path):
            raise ValueError("Filepath is not set or points to a remote URL")
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def write(self, filepath: None | str | Path = None, **kwargs) -> bool:
        """Write output to disk, filepath is not required if it was provided to parent App during init."""
        if filepath is None:
            return self.parent_pyotb_app.write(
                {self.output_image_key: self.filepath}, **kwargs
            )
        return self.parent_pyotb_app.write({self.output_image_key: filepath}, **kwargs)

    def __str__(self) -> str:
        """Return string representation of Output filepath."""
        return str(self.filepath)


def get_nbchannels(inp: str | Path | OTBObject) -> int:
    """Get the nb of bands of input image.

    Args:
        inp: can be filepath or OTBObject object

    Returns:
        number of bands in image

    """
    if isinstance(inp, OTBObject):
        return inp.shape[-1]
    if isinstance(inp, (str, Path)):
        # Executing the app, without printing its log
        try:
            info = App("ReadImageInfo", inp, quiet=True)
            return info["numberbands"]
        except (
            RuntimeError
        ) as info_err:  # this happens when we pass a str that is not a filepath
            raise TypeError(
                f"Could not get the number of channels file '{inp}' ({info_err})"
            ) from info_err
    raise TypeError(f"Can't read number of channels of type '{type(inp)}' object {inp}")


def get_pixel_type(inp: str | Path | OTBObject) -> str:
    """Get the encoding of input image pixels.

    Args:
        inp: can be filepath or pyotb object

    Returns:
        pixel_type: OTB enum e.g. `otbApplication.ImagePixelType_uint8', which actually is an int.
                    For an OTBObject with several outputs, only the pixel type of the first output is returned

    """
    if isinstance(inp, OTBObject):
        return inp.app.GetParameterOutputImagePixelType(inp.output_image_key)
    if isinstance(inp, (str, Path)):
        try:
            info = App("ReadImageInfo", inp, quiet=True)
            datatype = info["datatype"]  # which is such as short, float...
        except (
            RuntimeError
        ) as info_err:  # this happens when we pass a str that is not a filepath
            raise TypeError(
                f"Could not get the pixel type of `{inp}` ({info_err})"
            ) from info_err
        if datatype:
            return parse_pixel_type(datatype)
    raise TypeError(f"Could not get the pixel type of {type(inp)} object {inp}")


def parse_pixel_type(pixel_type: str | int) -> int:
    """Convert one str pixel type to OTB integer enum if necessary.

    Args:
        pixel_type: pixel type. can be str, int or dict

    Returns:
        pixel_type integer value

    """
    if isinstance(pixel_type, int):  # normal OTB int enum
        return pixel_type
    if isinstance(pixel_type, str):  # correspond to 'uint8' etc...
        datatype_to_pixeltype = {
            "unsigned_char": "uint8",
            "short": "int16",
            "unsigned_short": "uint16",
            "int": "int32",
            "unsigned_int": "uint32",
            "long": "int32",
            "ulong": "uint32",
            "float": "float",
            "double": "double",
        }
        if pixel_type in datatype_to_pixeltype.values():
            return getattr(otb, f"ImagePixelType_{pixel_type}")
        if pixel_type in datatype_to_pixeltype:
            return getattr(otb, f"ImagePixelType_{datatype_to_pixeltype[pixel_type]}")
        raise KeyError(
            f"Unknown data type `{pixel_type}`. Available ones: {datatype_to_pixeltype}"
        )
    raise TypeError(
        f"Bad pixel type specification ({pixel_type} of type {type(pixel_type)})"
    )


def get_out_images_param_keys(app: OTBObject) -> list[str]:
    """Return every output parameter keys of an OTB app."""
    return [
        key
        for key in app.GetParametersKeys()
        if app.GetParameterType(key) == otb.ParameterType_OutputImage
    ]


def summarize(
    obj: App | Output | str | float | list,
    strip_inpath: bool = False,
    strip_outpath: bool = False,
) -> dict[str, dict | Any] | str | float | list:
    """Recursively summarize parameters of an App or Output object and its parents.

    At the deepest recursion level, this function just return any parameter value,
     path stripped if needed, or app summarized in case of a pipeline.

    Args:
        obj: input object / parameter value to summarize
        strip_inpath: strip all input paths: If enabled, paths related to
            inputs are truncated after the first "?" character. Can be useful
            to remove URLs tokens (e.g. SAS or S3 credentials).
        strip_outpath: strip all output paths: If enabled, paths related
            to outputs are truncated after the first "?" character. Can be
            useful to remove extended filenames.

    Returns:
        nested dictionary containing name and parameters of an app and its parents

    """
    # This is the deepest recursion level
    if isinstance(obj, list):
        return [summarize(o) for o in obj]
    if isinstance(obj, Output):
        return summarize(obj.parent_pyotb_app)
    if not isinstance(obj, App):
        return obj

    def strip_path(param: str | Any):
        if isinstance(param, list):
            return [strip_path(p) for p in param]
        if not isinstance(param, str):
            return summarize(param)
        return param.split("?")[0]

    # Call / top level of recursion : obj is an App
    # We need to return parameters values, summarized if param is an App
    parameters = {}
    for key, param in obj.parameters.items():
        if strip_inpath and obj.is_input(key) or strip_outpath and obj.is_output(key):
            parameters[key] = strip_path(param)
        else:
            parameters[key] = summarize(param)
    return {"name": obj.app.GetName(), "parameters": parameters}
