"""<b>Measure Object Intensity</b> measures several intensity features for 
identified objects.
<hr>
Given an image with objects identified (e.g. nuclei or cells), this
module extracts intensity features for each object based on one or more
corresponding grayscale images. Measurements are recorded for each object.

<p>Intensity measurements are made for all combinations of the images
and objects entered. If you want only specific image/object measurements, you can
use multiple MeasureObjectIntensity modules for each group of measurements desired.</p>

<p>Note that for publication purposes, the units of
intensity from microscopy images are usually described as "Intensity
units" or "Arbitrary intensity units" since microscopes are not
calibrated to an absolute scale. Also, it is important to note whether
you are reporting either the mean or the integrated intensity, so specify
"Mean intensity units" or "Integrated intensity units" accordingly.</p>

<p>Keep in mind that the default behavior in CellProfiler is to rescale the
image intensity from 0 to 1 by dividing all pixels in the image by the
maximum possible intensity value. This "maximum possible" value
is defined by the "Set intensity range from" setting in <b>NamesAndTypes</b>;
see the help for that setting for more details.</p>

<h4>Available measurements</h4>
<ul><li><i>IntegratedIntensity:</i> The sum of the pixel intensities within an
 object.</li>
<li><i>MeanIntensity:</i> The average pixel intensity within an object.</li>
<li><i>StdIntensity:</i> The standard deviation of the pixel intensities within
 an object.</li>
<li><i>MaxIntensity:</i> The maximal pixel intensity within an object.</li>
<li><i>MinIntensity:</i> The minimal pixel intensity within an object.</li>
<li><i>IntegratedIntensityEdge:</i> The sum of the edge pixel intensities of an
 object.</li>
<li><i>MeanIntensityEdge:</i> The average edge pixel intensity of an object.</li>
<li><i>StdIntensityEdge:</i> The standard deviation of the edge pixel intensities
 of an object.</li>
<li><i>MaxIntensityEdge:</i> The maximal edge pixel intensity of an object.</li>
<li><i>MinIntensityEdge:</i> The minimal edge pixel intensity of an object.</li>
<li><i>MassDisplacement:</i> The distance between the centers of gravity in the
 gray-level representation of the object and the binary representation of
 the object.</li>
<li><i>LowerQuartileIntensity:</i> The intensity value of the pixel for which 25%
 of the pixels in the object have lower values.</li>
<li><i>MedianIntensity:</i> The median intensity value within the object</li>
<li><i>MADIntensity:</i> The median absolute deviation (MAD) value of the
intensities within the object. The MAD is defined as the median(|x<sub>i</sub> - median(x)|).</li>
<li><i>UpperQuartileIntensity:</i> The intensity value of the pixel for which 75%
 of the pixels in the object have lower values.</li>
<li><i>Location_CenterMassIntensity_X, Location_CenterMassIntensity_Y:</i> The
pixel (X,Y) coordinates of the intensity weighted centroid (= center of mass = first moment)
of all pixels within the object.</li>
<li><i>Location_MaxIntensity_X, Location_MaxIntensity_Y:</i> The pixel (X,Y) coordinates of
the pixel with the maximum intensity within the object.</li>
</ul>

See also <b>NamesAndTypes</b>, <b>MeasureImageIntensity</b>.
"""

import centrosome.outline as cpmo
import numpy as np
import scipy.ndimage as nd
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.filter import stretch

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
from cellprofiler.modules.identify import C_LOCATION
from cellprofiler.modules import measureobjectintensity as cpmmoi
import mlgoc.objectsml

INTENSITY = 'Intensity'
INTEGRATED_INTENSITY = 'IntegratedIntensity'
MEAN_INTENSITY = 'MeanIntensity'
STD_INTENSITY = 'StdIntensity'
MIN_INTENSITY = 'MinIntensity'
MAX_INTENSITY = 'MaxIntensity'
INTEGRATED_INTENSITY_EDGE = 'IntegratedIntensityEdge'
MEAN_INTENSITY_EDGE = 'MeanIntensityEdge'
STD_INTENSITY_EDGE = 'StdIntensityEdge'
MIN_INTENSITY_EDGE = 'MinIntensityEdge'
MAX_INTENSITY_EDGE = 'MaxIntensityEdge'
MASS_DISPLACEMENT = 'MassDisplacement'
LOWER_QUARTILE_INTENSITY = 'LowerQuartileIntensity'
MEDIAN_INTENSITY = 'MedianIntensity'
MAD_INTENSITY = 'MADIntensity'
UPPER_QUARTILE_INTENSITY = 'UpperQuartileIntensity'
LOC_CMI_X = 'CenterMassIntensity_X'
LOC_CMI_Y = 'CenterMassIntensity_Y'
LOC_MAX_X = 'MaxIntensity_X'
LOC_MAX_Y = 'MaxIntensity_Y'

ALL_MEASUREMENTS = [INTEGRATED_INTENSITY, MEAN_INTENSITY, STD_INTENSITY,
                        MIN_INTENSITY, MAX_INTENSITY, INTEGRATED_INTENSITY_EDGE,
                        MEAN_INTENSITY_EDGE, STD_INTENSITY_EDGE,
                        MIN_INTENSITY_EDGE, MAX_INTENSITY_EDGE,
                        MASS_DISPLACEMENT, LOWER_QUARTILE_INTENSITY,
                        MEDIAN_INTENSITY, MAD_INTENSITY, UPPER_QUARTILE_INTENSITY]
ALL_LOCATION_MEASUREMENTS = [LOC_CMI_X, LOC_CMI_Y, LOC_MAX_X, LOC_MAX_Y]

class MeasureObjectIntensityML(cpmmoi.MeasureObjectIntensity):

    module_name = "MeasureObjectIntensityML"
    variable_revision_number = 1
    category = "Measurement"

    def run(self, workspace):
        if self.show_window:
            workspace.display_data.col_labels = (
                "Image","Object","Feature","Mean","Median","STD")
            workspace.display_data.statistics = statistics = []
        for object_name in [obj.name for obj in self.objects]:
            objects = workspace.object_set.get_objects(object_name.value)
            if isinstance(objects, mlgoc.objectsml.ObjectsML):
                self.run_on_objectsml(workspace,object_name,objects)
            else:
                self.run_on_objects(workspace,object_name,objects)

    def run_on_objects(self,workspace,object_name,objects):
        nobjects = objects.count
        for image_name in [img.name for img in self.images]:
            image = workspace.image_set.get_image(image_name.value,
                                                  must_be_grayscale=True)
            img = image.pixel_data
            if image.has_mask:
                masked_image = img.copy()
                masked_image[~image.mask] = 0
            else:
                masked_image = img
            integrated_intensity = np.zeros((nobjects,))
            mean_intensity = np.zeros((nobjects,))
            std_intensity = np.zeros((nobjects,))
            min_intensity = np.zeros((nobjects,))
            max_intensity = np.zeros((nobjects,))
            for labels, lindexes in objects.get_labels():
                lindexes = lindexes[lindexes != 0]
                labels, img = cpo.crop_labels_and_image(labels, img)
                _, masked_image = cpo.crop_labels_and_image(labels, masked_image)
                outlines = cpmo.outline(labels)
                if image.has_mask:
                    _, mask = cpo.crop_labels_and_image(labels, image.mask)
                    masked_labels = labels.copy()
                    masked_labels[~mask] = 0
                    masked_outlines = outlines.copy()
                    masked_outlines[~mask] = 0
                else:
                    masked_labels = labels
                    masked_outlines = outlines

                lmask = masked_labels > 0 & np.isfinite(img)  # Ignore NaNs, Infs
                has_objects = np.any(lmask)
                if has_objects:
                    limg = img[lmask]
                    llabels = labels[lmask]
                    mesh_y, mesh_x = np.mgrid[0:masked_image.shape[0],
                                     0:masked_image.shape[1]]
                    mesh_x = mesh_x[lmask]
                    mesh_y = mesh_y[lmask]
                    lcount = fix(nd.sum(np.ones(len(limg)), llabels, lindexes))
                    integrated_intensity[lindexes - 1] = \
                        fix(nd.sum(limg, llabels, lindexes))
                    mean_intensity[lindexes - 1] = \
                        integrated_intensity[lindexes - 1] / lcount
                    std_intensity[lindexes - 1] = np.sqrt(
                        fix(nd.mean((limg - mean_intensity[llabels - 1]) ** 2,
                                    llabels, lindexes)))
                    min_intensity[lindexes - 1] = fix(nd.minimum(limg, llabels, lindexes))
                    max_intensity[lindexes - 1] = fix(
                        nd.maximum(limg, llabels, lindexes))
                    # Compute the position of the intensity maximum
                    max_position = np.array(fix(nd.maximum_position(limg, llabels, lindexes)), dtype=int)
                    max_position = np.reshape(max_position, (max_position.shape[0],))
                    # max_x[lindexes - 1] = mesh_x[max_position]
                    # max_y[lindexes - 1] = mesh_y[max_position]
                    # The mass displacement is the distance between the center
                    # of mass of the binary image and of the intensity image. The
                    # center of mass is the average X or Y for the binary image
                    # and the sum of X or Y * intensity / integrated intensity
                    cm_x = fix(nd.mean(mesh_x, llabels, lindexes))
                    cm_y = fix(nd.mean(mesh_y, llabels, lindexes))

                    i_x = fix(nd.sum(mesh_x * limg, llabels, lindexes))
                    i_y = fix(nd.sum(mesh_y * limg, llabels, lindexes))
                    # cmi_x[lindexes - 1] = i_x / integrated_intensity[lindexes - 1]
                    # cmi_y[lindexes - 1] = i_y / integrated_intensity[lindexes - 1]
                    # diff_x = cm_x - cmi_x[lindexes - 1]
                    # diff_y = cm_y - cmi_y[lindexes - 1]
                    # mass_displacement[lindexes - 1] = \
                    #     np.sqrt(diff_x * diff_x + diff_y * diff_y)
                    #
                    # Sort the intensities by label, then intensity.
                    # For each label, find the index above and below
                    # the 25%, 50% and 75% mark and take the weighted
                    # average.
                    #
                    order = np.lexsort((limg, llabels))
                    areas = lcount.astype(int)
                    indices = np.cumsum(areas) - areas
                    # for dest, fraction in (
                    #         (lower_quartile_intensity, 1.0 / 4.0),
                    #         (median_intensity, 1.0 / 2.0),
                    #         (upper_quartile_intensity, 3.0 / 4.0)):
                    #     qindex = indices.astype(float) + areas * fraction
                    #     qfraction = qindex - np.floor(qindex)
                    #     qindex = qindex.astype(int)
                    #     qmask = qindex < indices + areas - 1
                    #     qi = qindex[qmask]
                    #     qf = qfraction[qmask]
                    #     dest[lindexes[qmask] - 1] = (
                    #         limg[order[qi]] * (1 - qf) +
                    #         limg[order[qi + 1]] * qf)
                    #     #
                    #     # In some situations (e.g. only 3 points), there may
                    #     # not be an upper bound.
                    #     #
                    #     qmask = (~qmask) & (areas > 0)
                    #     dest[lindexes[qmask] - 1] = limg[order[qindex[qmask]]]
                    #
                    # Once again, for the MAD
                    #
                    # madimg = np.abs(limg - median_intensity[llabels - 1])
                    # order = np.lexsort((madimg, llabels))
                    qindex = indices.astype(float) + areas / 2.0
                    qfraction = qindex - np.floor(qindex)
                    qindex = qindex.astype(int)
                    qmask = qindex < indices + areas - 1
                    qi = qindex[qmask]
                    qf = qfraction[qmask]
                    # mad_intensity[lindexes[qmask] - 1] = (
                    #     madimg[order[qi]] * (1 - qf) +
                    #     madimg[order[qi + 1]] * qf)
                    # qmask = (~qmask) & (areas > 0)
                    # mad_intensity[lindexes[qmask] - 1] = madimg[order[qindex[qmask]]]
            m = workspace.measurements
            for category, feature_name, measurement in \
                    ((INTENSITY, INTEGRATED_INTENSITY, integrated_intensity), \
                     (INTENSITY, MEAN_INTENSITY, mean_intensity), \
                     (INTENSITY, STD_INTENSITY, std_intensity), \
                     (INTENSITY, MIN_INTENSITY, min_intensity), \
                     (INTENSITY, MAX_INTENSITY, max_intensity)):
                measurement_name = "%s_%s_%s" % (category, feature_name, image_name.value)
                m.add_measurement(object_name.value, measurement_name, measurement)

                if self.show_window and len(measurement) > 0:
                    workspace.display_data.statistics.append((image_name.value, object_name.value,
                                                              feature_name,
                                                              np.round(np.mean(measurement), 3),
                                                              np.round(np.median(measurement), 3),
                                                              np.round(np.std(measurement), 3)))

    def run_on_objectsml(self,workspace,object_name,objectsml):
        for image_name in [img.name for img in self.images]:
            image = workspace.image_set.get_image(image_name.value,
                                                  must_be_grayscale=True)
            img = image.pixel_data
            if image.has_mask:
                masked_image = img.copy()
                masked_image[~image.mask] = 0
            else:
                masked_image = img
            labels = objectsml.get_labels()
            unique_values = np.unique(labels)
            nobjects = len(unique_values) - 1

            integrated_intensity = np.zeros((nobjects,))
            mean_intensity = np.zeros((nobjects,))
            std_intensity = np.zeros((nobjects,))
            min_intensity = np.zeros((nobjects,))
            max_intensity = np.zeros((nobjects,))

            if len(unique_values) > 1:
                for labeli in range(len(unique_values) - 1):
                    label = unique_values[labeli + 1]
                    indices = np.where(labels == label)
                    if len(indices) == 2:  # 2D label matrix
                        intensities = img[indices]
                    else:
                        intensities = img[(indices[1], indices[2],)]
                    integrated_intensity[labeli] = sum(intensities)
                    mean_intensity[labeli] = integrated_intensity[labeli] / len(intensities)
                    std_intensity[labeli] = np.std(intensities)
                    min_intensity[labeli] = min(intensities)
                    max_intensity[labeli] = max(intensities)

            m = workspace.measurements
            for category, feature_name, measurement in \
                    ((INTENSITY, INTEGRATED_INTENSITY, integrated_intensity), \
                     (INTENSITY, MEAN_INTENSITY, mean_intensity), \
                     (INTENSITY, STD_INTENSITY, std_intensity), \
                     (INTENSITY, MIN_INTENSITY, min_intensity), \
                     (INTENSITY, MAX_INTENSITY, max_intensity)):
                measurement_name = "%s_%s_%s" % (category, feature_name, image_name.value)
                m.add_measurement(object_name.value, measurement_name, measurement)
                if self.show_window and len(measurement) > 0:
                    workspace.display_data.statistics.append((image_name.value, object_name.value,
                                                              feature_name,
                                                              np.round(np.mean(measurement), 3),
                                                              np.round(np.median(measurement), 3),
                                                              np.round(np.std(measurement), 3)))

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0,
                             workspace.display_data.statistics,
                             col_labels = workspace.display_data.col_labels)
