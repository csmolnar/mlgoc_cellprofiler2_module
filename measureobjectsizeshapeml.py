from cellprofiler.gui.help import MEASUREOBJSIZESHAPE_ECCENTRICITY
__doc__ = '''
<b>Measure Object Size Shape </b> measures several area and shape
features of identified objects.
<hr>
Given an image with identified objects (e.g. nuclei or cells), this
module extracts area and shape features of each one. Note that these
features are only reliable for objects that are completely inside the
image borders, so you may wish to exclude objects touching the edge of
the image using <b>IdentifyPrimaryObjects</b>.

<p>Please note that the display window for this module shows per-image aggregates for the per-object
measurements. If you want to view the per-object measurements themselves, you will need to use
<b>ExportToSpreadsheet</b> to export them, or use <b>DisplayDataOnImage</b> to display the
object measurements of choice overlaid on an image of choice.</p>

<h4>Available measurements</h4>
See the <i>Technical Notes</i> below for an explanation of creating an ellipse with the
same second-moments as an object region.

<ul>
<li><i>Area:</i> The actual number of pixels in the region.</li>
<li><i>Perimeter:</i> The total number of pixels around the boundary of each
region in the image.</li>
<li><i>FormFactor:</i> Calculated as 4*&pi;*Area/Perimeter<sup>2</sup>. Equals 1 for a
perfectly circular object.</li>
<li><i>Solidity:</i> The proportion of the pixels in the convex hull that
are also in the object, i.e. <i>ObjectArea/ConvexHullArea</i>. Equals 1 for a solid object
(i.e., one with no holes or has a concave boundary), or &lt;1 for an object
with holes or possessing a convex/irregular boundary.</li>
<li><i>Extent:</i> The proportion of the pixels in the bounding box that
are also in the region. Computed as the Area divided by the area of the
bounding box.</li>
<li><i>EulerNumber:</i> The number of objects in the region
minus the number of holes in those objects, assuming 8-connectivity.</li>
<li><i>Center_X, Center_Y:</i> The <i>x</i>- and <i>y</i>-coordinates of the
point farthest away from any object edge. Note that this is not the same as the
<i>Location-X</i> and <i>-Y</i> measurements produced by the <b>Identify</b>
modules.
</li>
<li><i>Eccentricity:</i> The eccentricity of the ellipse that has the
same second-moments as the region. The eccentricity is the ratio of the
distance between the foci of the ellipse and its major axis length. The
value is between 0 and 1. (0 and 1 are degenerate cases; an ellipse whose
eccentricity is 0 is actually a circle, while an ellipse whose eccentricity
is 1 is a line segment.)
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:%(MEASUREOBJSIZESHAPE_ECCENTRICITY)s"></td></tr>
</table></li>
<li><i>MajorAxisLength:</i> The length (in pixels) of the major axis of
the ellipse that has the same normalized second central moments as the
region.</li>
<li><i>MinorAxisLength:</i> The length (in pixels) of the minor axis of
the ellipse that has the same normalized second central moments as the
region.</li>
<li><i>Orientation:</i> The angle (in degrees ranging from -90 to 90
degrees) between the x-axis and the major axis of the ellipse that has the
same second-moments as the region.</li>
<li><i>Compactness:</i> The mean squared distance of the object's
pixels from the centroid divided by the area. A filled circle will have
a compactness of 1, with irregular objects or objects with holes having
a value greater than 1.</li>
<li><i>MaximumRadius:</i> The maximum distance of any pixel in the object
to the closest pixel outside of the object. For skinny objects, this
is 1/2 of the maximum width of the object.</li>
<li><i>MedianRadius:</i> The median distance of any pixel in the object
to the closest pixel outside of the object.</li>
<li><i>MeanRadius:</i> The mean distance of any pixel in the object
to the closest pixel outside of the object.</li>
<li><i>MinFeretDiameter, MaxFeretDiameter:</i> The Feret diameter is the
distance between two parallel lines tangent on either side of the object
(imagine taking a caliper and measuring the object at various angles).
The minimum and maximum Feret diameters are the smallest and largest possible
diameters, rotating the calipers along all possible angles.</li>
<li><i>Zernike shape features:</i> Measure shape by describing a binary object (or
more precisely, a patch with background and an object in the center) in a
basis of Zernike polynomials, using the coefficients as features (<i>Boland
et al., 1998</i>). Currently, Zernike polynomials from order 0 to order 9 are
calculated, giving in total 30 measurements. While there is no limit to
the order which can be calculated (and indeed users could add more by
adjusting the code), the higher order polynomials carry less information.</li>
</ul>

<h4>Technical notes</h4>
A number of the object measurements are generated by creating an ellipse with the
same second-moments as the original object region. This is essentially the
best-fitting ellipse for a given object with the same statistical properties.
Furthermore, they are not affected by the translation or uniform scaling of a region.

<p>The Zernike features are computed within the minimum enclosing circle
of the object, i.e., the circle of the smallest diameter that contains all of the
object's pixels.</p>

<h4>References</h4>
<ul>
<li>Rocha L, Velho L, Carvalho PCP, "Image moments-based structuring and tracking of objects",
Proceedings from XV Brazilian Symposium on Computer Graphics and Image Processing, 2002.
<a href="http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf">(pdf)</a>
</li>
<li>Principles of Digital Image Processing: Core Algorithms (Undergraduate Topics in Computer Science):
<a href="http://www.scribd.com/doc/58004056/Principles-of-Digital-Image-Processing#page=49">Section 2.4.3 - Statistical shape properties</a>
</li>
<li>Chrystal P (1885), "On the problem to construct the minimum circle enclosing n
given points in a plane", <i>Proceedings of the Edinburgh Mathematical Society</i>,
vol 3, p. 30 </li>
</ul>

See also <b>MeasureImageAreaOccupied</b>.
'''%globals()

import numpy as np
import scipy.ndimage as scind

import centrosome.zernike as cpmz
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import ellipse_from_second_moments_ijv
from centrosome.cpmorphology import calculate_extents
from centrosome.cpmorphology import calculate_perimeters
from centrosome.cpmorphology import calculate_solidity
from centrosome.cpmorphology import euler_number
from centrosome.cpmorphology import distance_to_edge
from centrosome.cpmorphology import maximum_position_of_labels
from centrosome.cpmorphology import median_of_labels
from centrosome.cpmorphology import feret_diameter
from centrosome.cpmorphology import convex_hull_ijv
from cellprofiler.modules import measureobjectsizeshape as cpmmoss
import mlgoc.objectsml

"""The category of the per-object measurements made by this module"""
AREA_SHAPE = 'AreaShape'

F_AREA = "Area"
F_ECCENTRICITY = 'Eccentricity'
F_SOLIDITY = 'Solidity'
F_EXTENT = 'Extent'
F_CENTER_X = 'Center_X'
F_CENTER_Y = 'Center_Y'
F_EULER_NUMBER = 'EulerNumber'
F_PERIMETER = 'Perimeter'
F_FORM_FACTOR = 'FormFactor'
F_MAJOR_AXIS_LENGTH = 'MajorAxisLength'
F_MINOR_AXIS_LENGTH = 'MinorAxisLength'
F_ORIENTATION = 'Orientation'
F_COMPACTNESS = 'Compactness'
F_MAXIMUM_RADIUS = 'MaximumRadius'
F_MEDIAN_RADIUS = 'MedianRadius'
F_MEAN_RADIUS = 'MeanRadius'
F_MIN_FERET_DIAMETER = 'MinFeretDiameter'
F_MAX_FERET_DIAMETER = 'MaxFeretDiameter'

"""The non-Zernike features"""
F_STANDARD = [F_AREA, F_ECCENTRICITY, F_SOLIDITY, F_EXTENT,
              F_EULER_NUMBER, F_PERIMETER, F_FORM_FACTOR,
              F_MAJOR_AXIS_LENGTH, F_MINOR_AXIS_LENGTH,
              F_ORIENTATION, F_COMPACTNESS, F_CENTER_X, F_CENTER_Y,
              F_MAXIMUM_RADIUS, F_MEAN_RADIUS, F_MEDIAN_RADIUS,
              F_MIN_FERET_DIAMETER, F_MAX_FERET_DIAMETER]

class MeasureObjectSizeShapeML(cpmmoss.MeasureObjectSizeShape):

    module_name = "MeasureObjectSizeShapeML"
    variable_revision_number = 1
    category = 'Measurement'

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""

        if self.show_window:
            workspace.display_data.col_labels = \
                     ("Object", "Feature", "Mean", "Median", "STD")
            workspace.display_data.statistics = []

        for object_group in self.object_groups:
            objects = workspace.object_set.get_objects(object_group.name.value)
            if isinstance(objects,mlgoc.objectsml.ObjectsML):
                self.run_on_objectsml(object_group.name.value, workspace, objects)
            else:
                self.run_on_objects(object_group.name.value, workspace)

    def run_on_objectsml(self, object_name, workspace, objectsml):
        """Run, computing the area measurements for a single map of objects"""
        #
        # Do the ellipse-related measurements
        #
        labels = objectsml.get_labels()
        unique_values = np.unique(labels)
        object_labels = unique_values[unique_values > 0]
        nobjects = len(object_labels)
        if nobjects > 0:
            object_indices = np.zeros((object_labels.max(),), dtype=object_labels.dtype)

            # Create objects in ijv format
            obji = []
            objj = []
            objl = []
            for ind,label in enumerate(object_labels):
                limg = labels[label-1,:,:]
                oi,oj = np.where(limg == label)
                ol = np.full(oi.shape, label, dtype=oi.dtype)
                obji.append(oi)
                objj.append(oj)
                objl.append(ol)
                object_indices[label-1] = ind

            i = np.concatenate(obji)
            j = np.concatenate(objj)
            l = np.concatenate(objl)
            
            centers, eccentricity, major_axis_length, minor_axis_length, \
                theta, compactness =\
                ellipse_from_second_moments_ijv(i, j, 1, l, object_labels, True)
            self.record_measurement(workspace, object_name,
                                    F_ECCENTRICITY, eccentricity)
            self.record_measurement(workspace, object_name,
                                    F_MAJOR_AXIS_LENGTH, major_axis_length)
            self.record_measurement(workspace, object_name,
                                    F_MINOR_AXIS_LENGTH, minor_axis_length)
            self.record_measurement(workspace, object_name, F_ORIENTATION,
                                    theta * 180 / np.pi)
            self.record_measurement(workspace, object_name, F_COMPACTNESS,
                                    compactness)

            mcenter_x = np.zeros(nobjects)
            mcenter_y = np.zeros(nobjects)
            mextent = np.zeros(nobjects)
            mperimeters = np.zeros(nobjects)
            msolidity = np.zeros(nobjects)
            euler = np.zeros(nobjects)
            max_radius = np.zeros(nobjects)
            median_radius = np.zeros(nobjects)
            mean_radius = np.zeros(nobjects)
            min_feret_diameter = np.zeros(nobjects)
            max_feret_diameter = np.zeros(nobjects)
            zernike_numbers = self.get_zernike_numbers()
            zf = {}
            for n,m in zernike_numbers:
                zf[(n,m)] = np.zeros(nobjects)

            ijv = np.zeros((i.shape[0],3),dtype=i.dtype)
            ijv[:,0] = i
            ijv[:,1] = j
            ijv[:,2] = l
            chulls, chull_counts = convex_hull_ijv(ijv, object_labels)
            for label in object_labels:
                to_indices = object_indices[label-1]
                limg = labels[to_indices,:,:]
                distance = distance_to_edge(limg)
                mcenter_y[to_indices], mcenter_x[to_indices] =\
                         maximum_position_of_labels(distance, limg, [label])
                max_radius[to_indices] = fix(scind.maximum(
                    distance, limg, [label]))
                mean_radius[to_indices] = fix(scind.mean(
                    distance, limg, [label]))
                median_radius[to_indices] = median_of_labels(
                    distance, limg, [label])
                #
                # The extent (area / bounding box area)
                #
                mextent[to_indices] = calculate_extents(limg, [label])
                #
                # The perimeter distance
                #
                mperimeters[to_indices] = calculate_perimeters(limg, [label])
                #
                # Solidity
                #
                msolidity[to_indices] = calculate_solidity(limg, [label])
                #
                # Euler number
                #
                euler[to_indices] = euler_number(limg, [label])
                #
                # Zernike features
                #
                zf_l = cpmz.zernike(zernike_numbers, limg, [label])
                for (n,m), z in zip(zernike_numbers, zf_l.transpose()):
                    zf[(n,m)][to_indices] = z
            #
            # Form factor
            #
            areas = np.bincount(np.reshape(labels, labels.size))
            areas = areas[1:]
            ff = 4.0 * np.pi * areas / mperimeters**2
            #
            # Feret diameter
            #
            min_feret_diameter, max_feret_diameter = \
                feret_diameter(chulls, chull_counts, object_labels)

            for f, m in ([(F_AREA, areas),
                          (F_CENTER_X, mcenter_x),
                          (F_CENTER_Y, mcenter_y),
                          (F_EXTENT, mextent),
                          (F_PERIMETER, mperimeters),
                          (F_SOLIDITY, msolidity),
                          (F_FORM_FACTOR, ff),
                          (F_EULER_NUMBER, euler),
                          (F_MAXIMUM_RADIUS, max_radius),
                          (F_MEAN_RADIUS, mean_radius),
                          (F_MEDIAN_RADIUS, median_radius),
                          (F_MIN_FERET_DIAMETER, min_feret_diameter),
                          (F_MAX_FERET_DIAMETER, max_feret_diameter)] +
                         [(self.get_zernike_name((n,m)), zf[(n,m)])
                          for n,m in zernike_numbers]):
                self.record_measurement(workspace, object_name, f, m)

        else:
            ff = np.zeros(0)