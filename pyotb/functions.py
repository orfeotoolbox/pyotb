"""This module provides a set of functions for pyotb."""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
import uuid
from collections import Counter
from pathlib import Path

from .core import App, Input, LogicalOperation, Operation, get_nbchannels
from .helpers import logger


def where(cond: App | str, x: App | str | float, y: App | str | float) -> Operation:
    """Functionally similar to numpy.where. Where cond is True (!=0), returns x. Else returns y.

    If cond is monoband whereas x or y are multiband, cond channels are expanded to match x & y ones.

    Args:
        cond: condition, must be a raster (filepath, App, Operation...).
        x: value if cond is True. Can be: float, int, App, filepath, Operation...
        y: value if cond is False. Can be: float, int, App, filepath, Operation...

    Returns:
        an output where pixels are x if cond is True, else y

    Raises:
        ValueError: if x and y have different number of bands

    """
    # Checking the number of bands of rasters. Several cases :
    # - if cond is monoband, x and y can be multibands. Then cond will adapt to match x and y nb of bands
    # - if cond is multiband, x and y must have the same nb of bands if they are rasters.
    x_nb_channels, y_nb_channels = None, None
    if not isinstance(x, (int, float)):
        x_nb_channels = get_nbchannels(x)
    if not isinstance(y, (int, float)):
        y_nb_channels = get_nbchannels(y)
    if x_nb_channels and y_nb_channels:
        if x_nb_channels != y_nb_channels:
            raise ValueError(
                "X and Y images do not have the same number of bands. "
                f"X has {x_nb_channels} bands whereas Y has {y_nb_channels} bands"
            )

    x_or_y_nb_channels = x_nb_channels if x_nb_channels else y_nb_channels
    cond_nb_channels = get_nbchannels(cond)
    if (
        cond_nb_channels != 1
        and x_or_y_nb_channels
        and cond_nb_channels != x_or_y_nb_channels
    ):
        raise ValueError(
            "Condition and X&Y do not have the same number of bands. Condition has "
            f"{cond_nb_channels} bands whereas X&Y have {x_or_y_nb_channels} bands"
        )
    # If needed, duplicate the single band binary mask to multiband to match the dimensions of x & y
    if cond_nb_channels == 1 and x_or_y_nb_channels and x_or_y_nb_channels != 1:
        logger.info(
            "The condition has one channel whereas X/Y has/have %s channels. Expanding number"
            " of channels of condition to match the number of channels of X/Y",
            x_or_y_nb_channels,
        )
    # Get the number of bands of the result
    out_nb_channels = x_or_y_nb_channels or cond_nb_channels

    return Operation("?", cond, x, y, nb_bands=out_nb_channels)


def clip(image: App | str, v_min: App | str | float, v_max: App | str | float):
    """Clip values of image in a range of values.

    Args:
        image: input raster, can be filepath or any pyotb object
        v_min: minimum value of the range
        v_max: maximum value of the range

    Returns:
        raster whose values are clipped in the range

    """
    if isinstance(image, (str, Path)):
        image = Input(image)
    return where(image <= v_min, v_min, where(image >= v_max, v_max, image))


def all(*inputs):  # pylint: disable=redefined-builtin
    """Check if value is different than 0 everywhere along the band axis.

    For only one image, this function checks that all bands of the image are True (i.e. !=0)
     and outputs a singleband boolean raster
    For several images, this function checks that all images are True (i.e. !=0) and outputs
     a boolean raster, with as many bands as the inputs

    Args:
        inputs: inputs can be 1) a single image or 2) several images, either passed as separate arguments
                or inside a list

    Returns:
        AND intersection

    """
    # If necessary, flatten inputs
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = inputs[0]
    # Add support for generator inputs (to have the same behavior as built-in `all` function)
    if (
        isinstance(inputs, tuple)
        and len(inputs) == 1
        and inspect.isgenerator(inputs[0])
    ):
        inputs = list(inputs[0])
    # Transforming potential filepaths to pyotb objects
    inputs = [Input(inp) if isinstance(inp, str) else inp for inp in inputs]

    # Checking that all bands of the single image are True
    if len(inputs) == 1:
        inp = inputs[0]
        if isinstance(inp, LogicalOperation):
            res = inp[:, :, 0]
        else:
            res = inp[:, :, 0] != 0
        for band in range(1, inp.shape[-1]):
            if isinstance(inp, LogicalOperation):
                res = res & inp[:, :, band]
            else:
                res = res & (inp[:, :, band] != 0)
        return res

    # Checking that all images are True
    if isinstance(inputs[0], LogicalOperation):
        res = inputs[0]
    else:
        res = inputs[0] != 0
    for inp in inputs[1:]:
        if isinstance(inp, LogicalOperation):
            res = res & inp
        else:
            res = res & (inp != 0)
    return res


def any(*inputs):  # pylint: disable=redefined-builtin
    """Check if value is different than 0 anywhere along the band axis.

    For only one image, this function checks that at least one band of the image is True (i.e. !=0) and outputs
    a single band boolean raster
    For several images, this function checks that at least one of the images is True (i.e. !=0) and outputs
    a boolean raster, with as many bands as the inputs

    Args:
        inputs: inputs can be 1) a single image or 2) several images, either passed as separate arguments
                or inside a list

    Returns:
        OR intersection

    """
    # If necessary, flatten inputs
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = inputs[0]
    # Add support for generator inputs (to have the same behavior as built-in `any` function)
    if (
        isinstance(inputs, tuple)
        and len(inputs) == 1
        and inspect.isgenerator(inputs[0])
    ):
        inputs = list(inputs[0])
    # Transforming potential filepaths to pyotb objects
    inputs = [Input(inp) if isinstance(inp, str) else inp for inp in inputs]

    # Checking that at least one band of the image is True
    if len(inputs) == 1:
        inp = inputs[0]
        if isinstance(inp, LogicalOperation):
            res = inp[:, :, 0]
        else:
            res = inp[:, :, 0] != 0

        for band in range(1, inp.shape[-1]):
            if isinstance(inp, LogicalOperation):
                res = res | inp[:, :, band]
            else:
                res = res | (inp[:, :, band] != 0)
        return res

    # Checking that at least one image is True
    if isinstance(inputs[0], LogicalOperation):
        res = inputs[0]
    else:
        res = inputs[0] != 0
    for inp in inputs[1:]:
        if isinstance(inp, LogicalOperation):
            res = res | inp
        else:
            res = res | (inp != 0)
    return res


def run_tf_function(func):
    """This function enables using a function that calls some TF operations, with pyotb object as inputs.

    For example, you can write a function that uses TF operations like this :
        ```python
        @run_tf_function
        def multiply(input1, input2):
            import tensorflow as tf
            return tf.multiply(input1, input2)

        # Then you can use the function like this :
        result = multiply(pyotb_object1, pyotb_object1)  # this is a pyotb object
        ```

    Args:
        func: function taking one or several inputs and returning *one* output

    Returns:
        wrapper: a function that returns a pyotb object

    Raises:
        SystemError: if OTBTF apps are missing

    """
    try:
        from .apps import (  # pylint: disable=import-outside-toplevel
            TensorflowModelServe,
        )
    except ImportError as err:
        raise SystemError(
            "Could not run Tensorflow function: failed to import TensorflowModelServe."
            "Check that you have OTBTF configured (https://github.com/remicres/otbtf#how-to-install)"
        ) from err

    def get_tf_pycmd(output_dir, channels, scalar_inputs):
        """Create a string containing all python instructions necessary to create and save the Keras model.

        Args:
            output_dir: directory under which to save the model
            channels: list of raster channels (int). Contain `None` entries for non-raster inputs
            scalar_inputs: list of scalars (int/float). Contain `None` entries for non-scalar inputs

        Returns:
            the whole string code for function definition + model saving

        """
        # Getting the string definition of the tf function (e.g. "def multiply(x1, x2):...")
        # Maybe not entirely foolproof, maybe we should use dill instead? but it would add a dependency
        func_def_str = inspect.getsource(func)
        func_name = func.__name__

        create_and_save_model_str = func_def_str
        # Adding the instructions to create the model and save it to output dir
        create_and_save_model_str += textwrap.dedent(
            f"""
            import tensorflow as tf

            model_inputs = []
            tf_inputs = []
            for channel, scalar_input in zip({channels}, {scalar_inputs}):
                if channel:
                    input = tf.keras.Input((None, None, channel))
                    tf_inputs.append(input)
                    model_inputs.append(input)
                else:
                    if isinstance(scalar_input, int):  # TF doesn't like mixing float and int
                        scalar_input = float(scalar_input)
                    tf_inputs.append(scalar_input)

            output = {func_name}(*tf_inputs)

            # Create and save the .pb model
            model = tf.keras.Model(inputs=model_inputs, outputs=output)
            model.save("{output_dir}")
            """
        )

        return create_and_save_model_str

    def wrapper(*inputs, tmp_dir="/tmp"):
        """For the user point of view, this function simply applies some TensorFlow operations to some rasters.

        Implicitly, it saves a .pb model that describe the TF operations, then creates an OTB ModelServe application
        that applies this .pb model to the inputs.

        Args:
            *inputs: a list of pyotb objects, filepaths or int/float numbers
            tmp_dir: directory where temporary models can be written (Default value = '/tmp')

        Returns:
            a pyotb object, output of TensorFlowModelServe

        """
        # Get infos about the inputs
        channels = []
        scalar_inputs = []
        raster_inputs = []
        for inp in inputs:
            try:
                # This is for raster input
                channel = get_nbchannels(inp)
                channels.append(channel)
                scalar_inputs.append(None)
                raster_inputs.append(inp)
            except TypeError:
                # This is for other inputs (float, int)
                channels.append(None)
                scalar_inputs.append(inp)

        # Create and save the model. This is executed **inside an independent process** because (as of 2022-03),
        # tensorflow python library and OTBTF are incompatible
        out_savedmodel = os.path.join(tmp_dir, f"tmp_otbtf_model_{uuid.uuid4()}")
        pycmd = get_tf_pycmd(out_savedmodel, channels, scalar_inputs)
        cmd_args = [sys.executable, "-c", pycmd]
        # TODO: remove subprocess execution since this issues has been fixed with OTBTF 4.0
        try:
            subprocess.run(
                cmd_args,
                env=os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.SubprocessError:
            logger.debug("Failed to call subprocess")
        if not os.path.isdir(out_savedmodel):
            logger.info("Failed to save the model")

        # Initialize the OTBTF model serving application
        model_serve = TensorflowModelServe(
            {
                "model.dir": out_savedmodel,
                "optim.disabletiling": "on",
                "model.fullyconv": "on",
            },
            n_sources=len(raster_inputs),
            frozen=True,
        )
        # Set parameters and execute
        for i, inp in enumerate(raster_inputs):
            model_serve.set_parameters({f"source{i + 1}.il": [inp]})
        model_serve.execute()
        # Possible ENH: handle the deletion of the temporary model ?

        return model_serve

    return wrapper


def define_processing_area(
    *args,
    window_rule: str = "intersection",
    pixel_size_rule: str = "minimal",
    interpolator: str = "nn",
    reference_window_input: dict = None,
    reference_pixel_size_input: str = None,
) -> list[App]:
    """Given several inputs, this function handles the potential resampling and cropping to same extent.

    WARNING: Not fully implemented / tested

    Args:
        *args: list of raster inputs. Can be str (filepath) or pyotb objects
        window_rule: Can be 'intersection', 'union', 'same_as_input', 'specify' (Default value = 'intersection')
        pixel_size_rule: Can be 'minimal', 'maximal', 'same_as_input', 'specify' (Default value = 'minimal')
        interpolator: Can be 'bco', 'nn', 'linear' (Default value = 'nn')
        reference_window_input: Required if window_rule = 'same_as_input' (Default value = None)
        reference_pixel_size_input: Required if pixel_size_rule = 'same_as_input' (Default value = None)

    Returns:
        list of in-memory pyotb objects with all the same resolution, shape and extent

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
    for inp in inputs:
        if isinstance(inp, str):  # this is for filepaths
            metadata = Input(inp).app.GetImageMetaData("out")
        elif isinstance(inp, App):
            metadata = inp.app.GetImageMetaData(inp.output_param)
        else:
            raise TypeError(f"Wrong input : {inp}")
        metadatas[inp] = metadata

    # Get a metadata of an arbitrary image. This is just to compare later with other images
    any_metadata = next(iter(metadatas.values()))
    # Checking if all images have the same projection
    if not all(
        metadata["ProjectionRef"] == any_metadata["ProjectionRef"]
        for metadata in metadatas.values()
    ):
        logger.warning(
            "All images may not have the same CRS, which might cause unpredictable results"
        )

    # Handling different spatial footprints
    # TODO: find possible bug - ImageMetaData is not updated when running an app
    #  cf https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2234. Should we use ImageOrigin instead?
    if not all(
        md["UpperLeftCorner"] == any_metadata["UpperLeftCorner"]
        and md["LowerRightCorner"] == any_metadata["LowerRightCorner"]
        for md in metadatas.values()
    ):
        # Retrieving the bounding box that will be common for all inputs
        if window_rule == "intersection":
            # The coordinates depend on the orientation of the axis of projection
            if any_metadata["GeoTransform"][1] >= 0:
                ulx = max(md["UpperLeftCorner"][0] for md in metadatas.values())
                lrx = min(md["LowerRightCorner"][0] for md in metadatas.values())
            else:
                ulx = min(md["UpperLeftCorner"][0] for md in metadatas.values())
                lrx = max(md["LowerRightCorner"][0] for md in metadatas.values())
            if any_metadata["GeoTransform"][-1] >= 0:
                lry = min(md["LowerRightCorner"][1] for md in metadatas.values())
                uly = max(md["UpperLeftCorner"][1] for md in metadatas.values())
            else:
                lry = max(md["LowerRightCorner"][1] for md in metadatas.values())
                uly = min(md["UpperLeftCorner"][1] for md in metadatas.values())

        elif window_rule == "same_as_input":
            ulx = metadatas[reference_window_input]["UpperLeftCorner"][0]
            lrx = metadatas[reference_window_input]["LowerRightCorner"][0]
            lry = metadatas[reference_window_input]["LowerRightCorner"][1]
            uly = metadatas[reference_window_input]["UpperLeftCorner"][1]
        elif window_rule == "specify":
            # When the user explicitly specifies the bounding box -> add some arguments in the function
            ...
        elif window_rule == "union":
            # When the user wants the final bounding box to be the union of all bounding box
            #  It should replace any 'outside' pixel by some NoData -> add `fillvalue` argument in the function
            ...

        # Applying this bounding box to all inputs
        bounds = (ulx, uly, lrx, lry)
        logger.info(
            "Cropping all images to extent Upper Left (%s, %s), Lower Right (%s, %s)",
            *bounds,
        )
        new_inputs = []
        for inp in inputs:
            try:
                params = {
                    "in": inp,
                    "mode": "extent",
                    "mode.extent.unit": "phy",
                    "mode.extent.ulx": ulx,
                    "mode.extent.uly": uly,
                    "mode.extent.lrx": lrx,
                    "mode.extent.lry": lry,
                }
                new_input = App("ExtractROI", params, quiet=True)
                new_inputs.append(new_input)
                # Potentially update the reference inputs for later resampling
                if str(inp) == str(reference_pixel_size_input):
                    # We use comparison of string because calling '=='
                    # on pyotb objects implicitly calls BandMathX application, which is not desirable
                    reference_pixel_size_input = new_input
            except RuntimeError as err:
                raise ValueError(
                    f"Cannot define the processing area for input {inp}"
                ) from err
        inputs = new_inputs
        # Update metadatas
        metadatas = {input: input.app.GetImageMetaData("out") for input in inputs}

    # Get a metadata of an arbitrary image. This is just to compare later with other images
    any_metadata = next(iter(metadatas.values()))
    # Handling different pixel sizes
    if not all(
        md["GeoTransform"][1] == any_metadata["GeoTransform"][1]
        and md["GeoTransform"][5] == any_metadata["GeoTransform"][5]
        for md in metadatas.values()
    ):
        # Retrieving the pixel size that will be common for all inputs
        if pixel_size_rule == "minimal":
            # selecting the input with the smallest x pixel size
            reference_input = min(
                metadatas, key=lambda x: metadatas[x]["GeoTransform"][1]
            )
        if pixel_size_rule == "maximal":
            # selecting the input with the highest x pixel size
            reference_input = max(
                metadatas, key=lambda x: metadatas[x]["GeoTransform"][1]
            )
        elif pixel_size_rule == "same_as_input":
            reference_input = reference_pixel_size_input
        elif pixel_size_rule == "specify":
            # When the user explicitly specify the pixel size -> add argument inside the function
            ...

        pixel_size = metadatas[reference_input]["GeoTransform"][1]

        # Perform resampling on inputs that do not comply with the target pixel size
        logger.info("Resampling all inputs to resolution: %s", pixel_size)
        new_inputs = []
        for inp in inputs:
            if metadatas[inp]["GeoTransform"][1] != pixel_size:
                superimposed = App(
                    "Superimpose",
                    inr=reference_input,
                    inm=inp,
                    interpolator=interpolator,
                )
                new_inputs.append(superimposed)
            else:
                new_inputs.append(inp)
        inputs = new_inputs
        metadatas = {inp: inp.app.GetImageMetaData("out") for inp in inputs}

    # Final superimposition to be sure to have the exact same image sizes
    image_sizes = {}
    for inp in inputs:
        if isinstance(inp, str):
            inp = Input(inp)
        image_sizes[inp] = inp.shape[:2]
    # Selecting the most frequent image size. It will be used as reference.
    most_common_image_size, _ = Counter(image_sizes.values()).most_common(1)[0]
    same_size_images = [
        inp
        for inp, image_size in image_sizes.items()
        if image_size == most_common_image_size
    ]

    # Superimposition for images that do not have the same size as the others
    new_inputs = []
    for inp in inputs:
        if image_sizes[inp] != most_common_image_size:
            superimposed = App(
                "Superimpose",
                inr=same_size_images[0],
                inm=inp,
                interpolator=interpolator,
            )
            new_inputs.append(superimposed)
        else:
            new_inputs.append(inp)

    return new_inputs
