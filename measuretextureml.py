"""
<b>Measure Texture</b> measures the degree and nature of textures within
objects (versus smoothness).
<hr>
This module measures the variations in grayscale images.  An object (or
entire image) without much texture has a smooth appearance; an
object or image with a lot of texture will appear rough and show a wide
variety of pixel intensities.

<p>This module can also measure textures of objects against grayscale images.
Any input objects specified will have their texture measured against <i>all</i> input
images specfied, which may lead to image-object texture combinations that are unneccesary.
If you do not want this behavior, use multiple <b>MeasureTexture</b> modules to
specify the particular image-object measures that you want.</p>

<h4>Available measurements</h4>
<ul>
<li><i>Haralick Features:</i> Haralick texture features are derived from the
co-occurrence matrix, which contains information about how image intensities in pixels with a
certain position in relation to each other occur together. <b>MeasureTexture</b>
can measure textures at different scales; the scale you choose determines
how the co-occurrence matrix is constructed.
For example, if you choose a scale of 2, each pixel in the image (excluding
some border pixels) will be compared against the one that is two pixels to
the right. <b>MeasureTexture</b> quantizes the image into eight intensity
levels. There are then 8x8 possible ways to categorize a pixel with its
scale-neighbor. <b>MeasureTexture</b> forms the 8x8 co-occurrence matrix
by counting how many pixels and neighbors have each of the 8x8 intensity
combinations.
<p>Thirteen measurements are then calculated for the image by performing
mathematical operations on the co-occurrence matrix (the formulas can be found
<a href="http://murphylab.web.cmu.edu/publications/boland/boland_node26.html">here</a>):
<ul>
<li><i>AngularSecondMoment:</i> Measure of image homogeneity. A higher value of this
feature indicates that the intensity varies less in an image. Has a value of 1 for a
uniform image.</li>
<li><i>Contrast:</i> Measure of local variation in an image. A high contrast value
indicates a high degree of local variation, and is 0 for a uniform image.</li>
<li><i>Correlation:</i> Measure of linear dependency of intensity values in an image.
For an image with large areas of similar intensities, correlation is much higher than
for an image with noisier, uncorrelated intensities. Has a value of 1 or -1 for a
perfectly positively or negatively correlated image.</li>
<li><i>Variance:</i> Measure of the variation of image intensity values. For an image
with uniform intensity, the texture variance would be zero.</li>
<li><i>InverseDifferenceMoment:</i> Another feature to represent image contrast. Has a
low value for inhomogeneous images, and a relatively higher value for homogeneous images.</li>
<li><i>SumAverage:</i> The average of the normalized grayscale image in the spatial
domain.</li>
<li><i>SumVariance:</i> The variance of the normalized grayscale image in the spatial
domain.</li>
<li><i>SumEntropy:</i> A measure of randomness within an image. </li>
<li><i>Entropy:</i> An indication of the complexity within an image. A complex image
produces a high entropy value.</li>
<li><i>DifferenceVariance:</i> The image variation in a normalized co-occurance matrix.</li>
<li><i>DifferenceEntropy:</i> Another indication of the amount of randomness in an image.</li>
<li><i>InfoMeas1</i></li>
<li><i>InfoMeas2</i></li>
</ul>
Each measurement is suffixed with the direction of the offset used between
pixels in the co-occurrence matrix:
<ul>
<li><i>0:</i> Horizontal</li>
<li><i>90:</i> Vertical</li>
<li><i>45:</i> Diagonal</li>
<li><i>135:</i> Anti-diagonal</li>
</ul>
</p>
</li>
<li>
<i>Gabor "wavelet" features:</i> These features are similar to wavelet features,
and they are obtained by applying so-called Gabor filters to the image. The Gabor
filters measure the frequency content in different orientations. They are very
similar to wavelets, and in the current context they work exactly as wavelets, but
they are not wavelets by a strict mathematical definition. The Gabor
features detect correlated bands of intensities, for instance, images of
Venetian blinds would have high scores in the horizontal orientation.</li>
</ul>

<h4>Technical notes</h4>

To calculate the Haralick features, <b>MeasureTexture</b> normalizes the
co-occurence matrix at the per-object level by basing the intensity levels of the
matrix on the maximum and minimum intensity observed within each object. This
is beneficial for images in which the maximum intensities of the objects vary
substantially because each object will have the full complement of levels.

<p><b>MeasureTexture</b> performs a vectorized calculation of the Gabor filter,
properly scaled to the size of the object being measured and covering all
pixels in the object. The Gabor filter can be calculated at a user-selected
number of angles by using the following algorithm to compute a score
at each scale using the Gabor filter:
<ul>
<li>Divide the half-circle from 0 to 180&deg; by the number of desired
angles. For instance, if the user chooses two angles, <b>MeasureTexture</b>
uses 0 and 90 &deg; (horizontal and vertical) for the filter
orientations. This is the &theta; value from the reference paper.</li>
<li>For each angle, compute the Gabor filter for each object in the image
at two phases separated by 90&deg; in order to account for texture
features whose peaks fall on even or odd quarter-wavelengths.</li>
<li>Multiply the image times each Gabor filter and sum over the pixels
in each object.</li>
<li>Take the square root of the sum of the squares of the two filter scores.
This results in one score per &theta;.</li>
<li>Save the maximum score over all &theta; as the score at the desired scale.</li>
</ul>
</p>

<h4>References</h4>
<ul>
<li>Haralick RM, Shanmugam K, Dinstein I. (1973), "Textural Features for Image
Classification" <i>IEEE Transaction on Systems Man, Cybernetics</i>,
SMC-3(6):610-621.
<a href="http://dx.doi.org/10.1109/TSMC.1973.4309314">(link)</a></li>
<li>Gabor D. (1946). "Theory of communication"
<i>Journal of the Institute of Electrical Engineers</i> 93:429-441.
<a href="http://dx.doi.org/10.1049/ji-3-2.1946.0074">(link)</a></li>
</ul>
"""

import numpy as np
import scipy.ndimage as scind
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.filter import gabor
from centrosome.haralick import Haralick, normalized_per_object

import cellprofiler.objects as cpo
from cellprofiler.modules import measuretexture as cpmmt
import mlgoc.objectsml

"""The category of the per-object measurements made by this module"""
TEXTURE = 'Texture'

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

F_GABOR = "Gabor"

H_HORIZONTAL = "Horizontal"
A_HORIZONTAL = "0"
H_VERTICAL = "Vertical"
A_VERTICAL = "90"
H_DIAGONAL = "Diagonal"
A_DIAGONAL = "45"
H_ANTIDIAGONAL = "Anti-diagonal"
A_ANTIDIAGONAL = "135"
H_ALL = [H_HORIZONTAL, H_VERTICAL, H_DIAGONAL, H_ANTIDIAGONAL]

H_TO_A = { H_HORIZONTAL: A_HORIZONTAL,
           H_VERTICAL: A_VERTICAL,
           H_DIAGONAL: A_DIAGONAL,
           H_ANTIDIAGONAL: A_ANTIDIAGONAL }

class MeasureTextureML(cpmmt.MeasureTexture):

    module_name = "MeasureTextureML"
    variable_revision_number = 4
    category = 'Measurement'

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""

        workspace.display_data.col_labels = [
            "Image", "Object", "Measurement", "Scale", "Value"]
        statistics = []
        for image_group in self.image_groups:
            image_name = image_group.image_name.value
            for scale_group in self.scale_groups:
                scale = scale_group.scale.value
                if self.wants_image_measurements():
                    if self.wants_gabor:
                        statistics += self.run_image_gabor(
                            image_name, scale, workspace)
                    for angle in scale_group.angles.get_selections():
                        statistics += self.run_image(
                            image_name, scale, angle, workspace)
                if self.wants_object_measurements():
                    for object_group in self.object_groups:
                        object_name = object_group.object_name.value
                        objects = workspace.object_set.get_objects(object_name)
                        if isinstance(objects, mlgoc.objectsml.ObjectsML):
                            for angle in scale_group.angles.get_selections():
                                statistics += self.run_oneml(image_name, object_name, scale, angle, workspace)
                        else:
                            for angle in scale_group.angles.get_selections():
                                statistics += self.run_one(image_name, object_name, scale, angle, workspace)

                        if self.wants_gabor:
                            if isinstance(objects, mlgoc.objectsml.ObjectsML):
                                statistics += self.run_one_gaborml(
                                    image_name, object_name, scale, workspace)
                            else:
                                statistics += self.run_one_gabor(
                                    image_name, object_name, scale, workspace)
                        
        if self.show_window:
            workspace.display_data.statistics = statistics

    def run_oneml(self, image_name, object_name, scale, angle, workspace):
        """Run, computing the area measurements for a single map of objects"""
        statistics = []
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objectsml = workspace.object_set.get_objects(object_name)
        labels = objectsml.get_labels()
        unique_values = np.unique(labels)
        object_labels = unique_values[unique_values > 0]
        nobjects = len(object_labels)
        pixel_data = image.pixel_data
        if image.has_mask:
            mask = image.mask
        else:
            mask = None

        try:
            pixel_data = objectsml.crop_image_similarly(pixel_data)
        except ValueError:
            #
            # Recover by cropping the image to the labels
            #
            pixel_data, m1 = cpo.size_similarly(labels, pixel_data)
            if np.any(~m1):
                if mask is None:
                    mask = m1
                else:
                    mask, m2 = cpo.size_similarly(labels, mask)
                    mask[~m2] = False

        if np.all(labels == 0):
            for name in F_HARALICK:
                statistics += self.record_measurement(
                    workspace, image_name, object_name,
                    str(scale) + "_" + H_TO_A[angle], name, np.zeros((0,)))
        else:
            scale_i, scale_j = self.get_angle_ij(angle, scale)
            values = {}
            for name in F_HARALICK: values[name] = np.zeros((nobjects,))
            for layer in range(nobjects):
                limg = np.copy(labels[layer,:,:])
                limg[limg > 0] = 1
                for name, value in zip(F_HARALICK, Haralick(pixel_data,
                                                            limg,
                                                            scale_i,
                                                            scale_j,
                                                            mask=mask).all()):
                    values[name][layer] = value

            for name in F_HARALICK:
                statistics += self.record_measurement(
                    workspace, image_name, object_name,
                    str(scale) + "_" + H_TO_A[angle], name, values[name])

        return statistics

    def run_one_gaborml(self, image_name, object_name, scale, workspace):
        objectsml = workspace.object_set.get_objects(object_name)
        labels = objectsml.get_labels()
        unique_values = np.unique(labels)
        object_labels = unique_values[unique_values > 0]
        nobjects = len(object_labels)
        if nobjects > 0:
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data
            if image.has_mask:
                mask = image.mask
            else:
                mask = None
            try:
                pixel_data = objectsml.crop_image_similarly(pixel_data)
                if mask is not None:
                    mask = objectsml.crop_image_similarly(mask)
                    labels[~mask] = 0
            except ValueError:
                pixel_data, m1 = cpo.size_similarly(labels, pixel_data)
                labels[~m1] = 0
                if mask is not None:
                    mask, m2 = cpo.size_similarly(labels, mask)
                    labels[~m2] = 0
                    labels[~mask] = 0

            pixel_data = np.zeros(labels.shape, dtype=pixel_data.dtype)
            for layer in range(nobjects):
                limg = np.copy(labels[layer])
                limg[limg > 0] = 1
                pixel_data[layer] = normalized_per_object(pixel_data[layer], limg)
            best_score = np.zeros((nobjects,))
            for angle in range(self.gabor_angles.value):
                theta = np.pi * angle / self.gabor_angles.value
                scores = np.zeros((nobjects,))
                for layer in range(nobjects):
                    limg = np.copy(labels[layer])
                    limg[limg > 0] = 1
                    g = gabor(pixel_data[layer], limg, scale, theta)
                    score_r = fix(scind.sum(g.real, limg,
                                            np.arange(1, dtype=np.int32)+ 1))
                    score_i = fix(scind.sum(g.imag, limg,
                                            np.arange(1, dtype=np.int32)+ 1))
                    score = np.sqrt(score_r**2+score_i**2)
                    scores[layer] = score
                best_score = np.maximum(best_score, scores)
        else:
            best_score = np.zeros((0,))

        statistics = self.record_measurement(workspace,
                                             image_name,
                                             object_name,
                                             scale,
                                             F_GABOR,
                                             best_score)
        return statistics
