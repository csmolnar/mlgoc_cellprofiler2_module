'''<b>Convert Objects To Image </b> converts objects you have identified into an image.
<hr>

This module allows you to take previously identified objects and convert
them into an image according to a colormap you select, which can then be saved
with the <b>SaveImages</b> modules.

<p>If you would like to save your objects but do not need a colormap,
you can by bypass this module and use the <b>SaveImages</b> module directly
by specifying "Objects" as the type of image to save.
'''

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
from cellprofiler.modules import convertobjectstoimage as cpmcoi
import mlgoc.objectsml

DEFAULT_COLORMAP = "Default"
COLORCUBE = "colorcube"
LINES = "lines"
WHITE = "white"
COLORMAPS = ["Default", "autumn", "bone", COLORCUBE, "cool", "copper",
             "flag", "gray", "hot", "hsv", "jet", LINES,"pink", "prism",
             "spring", "summer", WHITE, "winter" ]

IM_COLOR = "Color"
IM_BINARY = "Binary (black & white)"
IM_GRAYSCALE = "Grayscale"
IM_UINT16 = "uint16"
IM_ALL = [IM_COLOR, IM_BINARY, IM_GRAYSCALE, IM_UINT16]


class ConvertObjectsToImageML(cpmcoi.ConvertObjectsToImage):

    module_name = "ConvertObjectsToImageML"
    category = "Object Processing"
    variable_revision_number = 1

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        alpha = np.zeros(objects.shape)
        if self.image_mode == IM_BINARY:
            pixel_data = np.zeros(objects.shape, bool)
        elif self.image_mode == IM_GRAYSCALE:
            pixel_data = np.zeros(objects.shape)
        elif self.image_mode == IM_UINT16:
            pixel_data = np.zeros(objects.shape, np.int32)
        else:
            pixel_data = np.zeros((objects.shape[0], objects.shape[1], 3))
        convert = True
        if isinstance(objects, mlgoc.objectsml.ObjectsML):
            labels = objects.get_labels()
            assert self.image_mode != IM_COLOR, "Conversion of multi-layered objects to colour image is not implemented"
            if self.image_mode == IM_BINARY:
                for l in range(objects.shape[0]):
                    layer = labels[l, :, :]>0
                    mask = layer != 0
                    pixel_data[l, :, :] = layer
                    alpha[l, :, :] = int(mask)
            elif self.image_mode == IM_GRAYSCALE:
                for l in range(objects.shape[0]):
                    layer = labels[l, :, :]
                    mask = layer != 0
                    layer = layer.astyp(float) / np.max(layer)
                    pixel_data[l, :, :] = layer[mask]
                    alpha[l, :, :] = int(mask)
            # elif self.image_mode == IM_COLOR:
            #     pass
            elif self.image_mode == IM_UINT16:
                for l in range(objects.shape[0]):
                    layer = labels[l, :, :]
                    mask = layer != 0
                    layer = layer.astype(float) / np.max(layer)
                    layer[mask] = 0
                    pixel_data[l, :, :] = layer
                    alpha[l, :, :] = mask.astype(np.float64)
                    convert = False
            mask = alpha > 0
            if self.image_mode == IM_BINARY:
                pass
            # elif self.image_mode == IM_COLOR:
            #     pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, np.newaxis]
            else:
                pixel_data[mask] = pixel_data[mask] / alpha[mask]
            image = cpi.Image(pixel_data, parent_image=objects.parent_image,
                              convert=convert)
            workspace.image_set.add(self.image_name.value, image)
            if self.show_window:
                workspace.display_data.objectml = True
                workspace.display_data.colour_image = labels
                workspace.display_data.pixel_data = pixel_data
        else:
            for labels, indices in objects.get_labels():
                mask = labels != 0
                if np.all(~ mask):
                    continue
                if self.image_mode == IM_BINARY:
                    pixel_data[mask] = True
                    alpha[mask] = 1
                elif self.image_mode == IM_GRAYSCALE:
                    pixel_data[mask] = labels[mask].astype(float) / np.max(labels)
                    alpha[mask] = 1
                elif self.image_mode == IM_COLOR:
                    import matplotlib.cm
                    from cellprofiler.gui.cpfigure_tools import renumber_labels_for_display
                    if self.colormap.value == DEFAULT_COLORMAP:
                        cm_name = cpprefs.get_default_colormap()
                    elif self.colormap.value == COLORCUBE:
                        # Colorcube missing from matplotlib
                        cm_name = "gist_rainbow"
                    elif self.colormap.value == LINES:
                        # Lines missing from matplotlib and not much like it,
                        # Pretty boring palette anyway, hence
                        cm_name = "Pastel1"
                    elif self.colormap.value == WHITE:
                        # White missing from matplotlib, it's just a colormap
                        # of all completely white... not even different kinds of
                        # white. And, isn't white just a uniform sampling of
                        # frequencies from the spectrum?
                        cm_name = "Spectral"
                    else:
                        cm_name = self.colormap.value
                    cm = matplotlib.cm.get_cmap(cm_name)
                    mapper = matplotlib.cm.ScalarMappable(cmap=cm)
                    pixel_data[mask, :] += \
                        mapper.to_rgba(renumber_labels_for_display(labels))[mask, :3]
                    alpha[mask] += 1
                elif self.image_mode == IM_UINT16:
                    pixel_data[mask] = labels[mask]
                    alpha[mask] = 1
                    convert = False
                mask = alpha > 0
                if self.image_mode == IM_BINARY:
                    pass
                elif self.image_mode == IM_COLOR:
                    pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, np.newaxis]
                else:
                    pixel_data[mask] = pixel_data[mask] / alpha[mask]
                image = cpi.Image(pixel_data, parent_image = objects.parent_image,
                                  convert = convert)
                workspace.image_set.add(self.image_name.value, image)
                if self.show_window:
                    workspace.display_data.objectml = False
                    workspace.display_data.ijv = objects.ijv
                    workspace.display_data.pixel_data = pixel_data

    def display(self, workspace, figure):
        pixel_data = workspace.display_data.pixel_data
        if workspace.display_data.objectml:
            pass
        else:
            figure.set_subplots((2, 1))
            figure.subplot_imshow_ijv(
                0, 0, workspace.display_data.ijv,
                shape = workspace.display_data.pixel_data.shape[:2],
                    title = "Original: %s"%self.object_name.value)
            if self.image_mode == IM_BINARY:
                figure.subplot_imshow_bw(1, 0, pixel_data,
                                                self.image_name.value,
                                                sharexy = figure.subplot(0, 0))
            elif pixel_data.shape[1] == 2:
                figure.subplot_imshow_grayscale(1, 0, pixel_data,
                                                self.image_name.value,
                                                sharexy = figure.subplot(0, 0))
            else:
                figure.subplot_imshow_grayscale(1, 0, pixel_data,
                                                self.image_name.value,
                                                sharexy = figure.subplot(0, 0))

    def create_colour_class_image(self,labels):
        # TODO implement visualization of overlapping
        pass
#
# Backwards compatability
#
# ConvertToImageMLGOC = ConvertObjectsToImageML
