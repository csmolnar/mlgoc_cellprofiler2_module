__doc__ = '''
<b>Identify Primary Objects MLGOC</b> Implementation of 'multi-layered gas of circles' model to segment near circular objects. See also Identify primary objects and Identify secondary objects modules.
<hr>
<h4>What do the settings mean?</h4>
See below for help on the individual settings.

<h4>What do I get as output?</h4>
A set of primary objects are produced by this module, which can be used in downstream modules
for measurement purposes or other operations. 
See the section <a href="#Available_measurements">"Available measurements"</a> below for 
the measurements that are produced by this module.

Once the module has finished processing, the module display window 
will show the following panels:
<ul>
<li><i>Upper left:</i> The raw, original image.</li>
<li><i>Upper right:</i> The segmented input objects as colored image.
It is important to note that assigned colors are arbitrary; 
they are used simply to help you distinguish the various objects. </li>
<li><i>Lower left:</i> The raw image overlaid with the colored outlines of objects.
Different colors denote different layers of the multi-layered model.</li>
<li><i>Lower right:</i> The raw image overlaid with the colored outlines of objects after postprocessing steps.</li>
Different colors denote different layers of the multi-layered model.
</ul>

<b>Object measurements:</b>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the primary 
object centroids.</li>
</ul>

<h4>Technical notes</h4>

The module first distributes the input objects between the layers of the multi-layered 'gas of near-circles' by applying
4 colouring algorithm for map coloring. This provides that close objects will lie in different layers.

<h4>References</h4>
<ul>
<li>Molnar, C., et. al. Accurate Morphology Preserving Segmentation of Overlapping Cells based on Active Contours. 
SciRep 2016.
(<a href="https://www.nature.com/articles/srep32412">link</a>)</li>
</ul>

<p>See also <b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b></p>
'''%globals()

# Developed by Csaba Molnar 2017
#
# Based on CellProfiler module identifyprimaryobjects.py and module
# IdentifyPrimaryMLGOC.m written for CellProfiler 1.0 by Csaba Molnar

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2015 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org


import scipy.ndimage
import numpy as np
import scipy.stats

from cellprofiler.modules import identify as cpmi
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import mlgoc.compute_mlgoc_parameters as cmp
import mlgoc.mlgoc_segmentation_gm as mlgoc
import centrosome.outline
import cellprofiler.preferences as cpp

import mlgoc.objectsml as objectsml
from cellprofiler.modules.identify import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, M_NUMBER_OBJECT_NUMBER
from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES, RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP

INIT_MODE_SEEDS_MANUAL = "Seeds (manual)"
INIT_MODE_SEEDS_CIRCULAR_MANUAL = "Circular seeds (manual)"
INIT_MODE_NEUTRAL = "Neutral"
INIT_MODE_SQUARES = "Squares"


LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

# Settings text which is referenced in various places in the help


class IdentifyPrimaryObjectsMLGOC(cpmi.Identify):
            
    variable_revision_number = 1
    category =  "Object Processing"
    module_name = "IdentifyPrimaryObjectsMLGOC"

    def create_settings(self):
        
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image", doc="""
            Select the image that you want to use to identify objects.""")

        self.seed_objects = cps.ObjectNameSubscriber(
            "Select input/seed objects", "Nuclei", doc="""
            What did you call the initial seeds for correction? Only used in case of manual initialization.""")

        self.object_name = cps.ObjectNameProvider(
            "Name the primary objects to be identified",
            "CorrectedNuclei", doc="""
            Enter the name that you want to call the objects identified by this module.""")

        self.preferred_radius = cps.Integer(
            "Preferred radius in pixels", 20, doc="""
            Give the approximate radius of object to identify.""")

        self.number_of_layers = cps.Integer(
            "Number of layers", 4, doc="""
            Give the number of overlapping layers. The default number is 4.""")

        self.mu_in = cps.Float(
            "Mean intensity of a single objects", 0.75, doc="""
            Give a mean intensity of a single (not overlapping) object.""")

        self.sigma_in = cps.Float(
            "Variance of intensities over single objects", 0.05, doc="""
            Give the variance of the intensities over single (not overlapping) objects.""")

        self.mu_out = cps.Float(
            "Mean intensity of the background", 0.25, doc="""
            Give a mean intensity the image background.""")

        self.sigma_out = cps.Float(
            "Variance of intensities over background", 0.05, doc="""
            Give the variance of the intensities over the background.""")

        self.data_weight = cps.Float(
            "Weight of data term", 1.0, doc="""
            Give the weight of data term compared to the shape energy.""")

        self.initialization_type = cps.Choice(
            "Type of initialization", [INIT_MODE_SEEDS_MANUAL, INIT_MODE_SEEDS_CIRCULAR_MANUAL, INIT_MODE_NEUTRAL], doc="""
            Select how to initialize the phase field.""")

        self.maximum_iterations = cps.Integer(
            "Maximum number of iterations", 200, doc="""
            Give the iteration number of the gradient descent optimization.""")

        self.exclude_border_objects = cps.Binary(
            "Discard objects touching the border of the image?", True, doc="""
            Removing objects that touch the image border.""")

        self.overlap_ji_threshold = cps.Float(
            "Threshold for degree of overlapping", 0.0, doc="""
            Discard overlapping objects if the Jaccard index of the overlapping area and the smaller object is over the given threshold.""")

        self.should_save_outlines = cps.Binary(
            'Retain outlines of the identified objects?', False, doc="""
             %(RETAINING_OUTLINES_HELP)s""" % globals())

        self.save_outlines = cps.OutlineNameProvider(
            'Name the outline image', "PrimaryOutlines", doc="""
             %(NAMING_OUTLINES_HELP)s""" % globals())

    def settings(self):
        return [self.image_name, self.seed_objects, self.object_name,
                self.preferred_radius, self.number_of_layers,
                self.mu_in, self.sigma_in, self.mu_out, self.sigma_out,
                self.data_weight, self.initialization_type,
                self.maximum_iterations, self.exclude_border_objects,
                self.overlap_ji_threshold,
                self.should_save_outlines, self.save_outlines]
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        """
        Upgrade the strings in setting_values dependent on saved revision
        """
        return setting_values, variable_revision_number, from_matlab
            
    def help_settings(self):
        return [self.image_name, self.seed_objects, self.object_name,
                self.preferred_radius, self.number_of_layer,
                self.mu_in, self.sigma_in, self.mu_out, self.sigma_out,
                self.data_weight, self.initialization_type,
                self.maximum_iterations, self.exclude_border_objects,
                self.overlap_ji_threshold,
                self.should_save_outlines, self.save_outlines]

    def visible_settings(self):
        vv = [self.image_name, self.seed_objects, self.object_name,
                self.preferred_radius, self.number_of_layers,
                self.mu_in, self.sigma_in, self.mu_out, self.sigma_out,
                self.data_weight, self.initialization_type,
                self.maximum_iterations, self.exclude_border_objects,
                self.overlap_ji_threshold]
        vv += [self.should_save_outlines]
        if self.should_save_outlines.value:
            vv += [self.save_outlines]
        return vv
    
    def run(self, workspace):
        """
        Run the module
        workspace    - contains
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
        """

        image_name = self.image_name.value
        cpimage = workspace.image_set.get_image(image_name,
                                                must_be_grayscale = True)
        image = cpimage.pixel_data
        mask = cpimage.mask

        objects_in = workspace.object_set.get_objects(self.seed_objects.value)
        labels_in = objects_in.segmented.copy()
        labels_in_mask = labels_in>0

        alpha_tilde = 1.0
        lambda_tilde = 1.0
        rhatstar = 1.0
        kappa = 0.0

        p = cmp.compute_mlgoc_parameters(alpha_tilde, lambda_tilde, self.preferred_radius.value, rhatstar)

        prior_phase_field_parameters = [p[1]] * self.number_of_layers.value

        gradient_weight = 0.0

        data_parameters = {'muin': self.mu_in.value,
                           'muout': self.mu_out.value,
                           'sigmain': self.sigma_in.value,
                           'sigmaout': self.sigma_out.value,
                           'gamma1': gradient_weight,
                           'gamma2': self.data_weight.value}

        maxd = int(max(map(lambda x: x['d'], prior_phase_field_parameters)))

        extended_image = np.pad(image,
                                ((2*maxd, 2*maxd), (2*maxd, 2*maxd)),
                                'constant', constant_values=(self.mu_out.value,))
        extended_image_height = extended_image.shape[0]
        extended_image_width = extended_image.shape[1]

        extended_image[extended_image>(data_parameters['muout']+4*(data_parameters['muin']-data_parameters['muout']))] =\
            data_parameters['muout']+4*(data_parameters['muin']-data_parameters['muout'])

        # initialize phase field
        if 'manual'.lower() in self.initialization_type.value.lower():
            # TODO: "labels_in, n = split_objects(labels_in,max_label,radius)"
            initial_phi = self.sort_grayscale_objects_to_layers_coloring(labels_in*labels_in_mask, self.number_of_layers.value)
            if self.initialization_type.value == INIT_MODE_SEEDS_CIRCULAR_MANUAL:
                raise NotImplementedError(("Initialization type %s is not yet supported"%
                                       (self.initialization_type.value)))
            initial_phi = np.pad(initial_phi,
                                 ((0,0),(2*maxd,2*maxd),(2*maxd,2*maxd)),
                                 'constant', constant_values=(-1.0,))
        else:
            initial_phi = np.array(np.random.normal(0, 0.1,
                                (self.number_of_layers.value, extended_image_height, extended_image_width)),ndmin=3)

        optimization_parameters = {'tolerance': 1e-10,
                                   'max_iterations': self.maximum_iterations.value,
                                   'save_frequency': -1}

        final_phi = mlgoc.mlgoc_segmentation_gm(
            prior_phase_field_parameters,
            extended_image,
            data_parameters,
            kappa,
            initial_phi,
            optimization_parameters)

        # label multi-layered segmentation
        labeled_image, object_count = scipy.ndimage.label(final_phi > p[1]['alpha']/p[1]['lambda'],
            structure=[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        if self.should_save_outlines.value:
            outlines = np.array(
                [centrosome.outline.outline(labeled_image[i, :, :]) for i in range(self.number_of_layers.value)])
            outline_image = np.any(outlines > 0, axis=0)
            out_img = cpi.Image(outline_image.astype(bool), parent_image=image)
            workspace.image_set.add(self.save_outlines.value, out_img)

        # TODO: remove embedded objects
        labeled_image = self.filter_on_embedded(labeled_image)
        original_segmentation = labeled_image.copy()  # before any filtering

        # TODO: remove small objects
        # size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = \
            self.filter_on_size(labeled_image, object_count)
        # size_excluded_labeled_image[labeled_image>0] = 0 # to store outlines of discarded objects on size
        # TODO: merge overlapping objects if JI>0.8
        labeled_image = self.merge_overlapping_objects(labeled_image, 0.8, self.preferred_radius.value)
        unedited_labels = labeled_image.copy()

        # TODO: remove overlapping objects (setting of module)
        # TODO: remove objects that are touching the border (setting)

        # bwlabel

        filtered_image = labeled_image.copy()

        outobjects = objectsml.ObjectsML()
        outobjects.segmented = labeled_image
        outobjects.unedited_segmented = unedited_labels
        outobjects.parent_image = cpimage
        workspace.object_set.add_objects(outobjects,self.object_name.value)

        object_count = len(np.unique(labeled_image))-1
        cpmi.add_object_count_measurements(workspace.measurements,
                                           self.object_name.value, object_count)
        add_object_location_measurements_ml(workspace.measurements,
                                              self.object_name.value,
                                              labeled_image)
        workspace.display_data.statistics = []

        if self.show_window:
            workspace.display_data.image = image
            workspace.display_data.mask = mask
            workspace.display_data.labels_in = labels_in
            workspace.display_data.labeled_image = original_segmentation
            workspace.display_data.filtered_image = filtered_image

    # def initialize_phi(self, workspace, extended_image_height, extended_image_width):
    #     if 'manual'.lower() in self.initialization_type.lower():
    #         labels_in = workspace.object_set.get_objects(self.seed_objects.value)
    #         if self.initialization_type == 'Seeds (manual)':
    #
    #             # TODO: insert distribution code here
    #             pass
    #         elif self.initialization_type == 'Circular seeds (manual)':
    #
    #             # TODO: insert distribution code here
    #             pass
    #         initial_phi = np.pad(initial_phi,((0,0),(),()),'constant',constant_values=-1.0)
    #     else:
    #         pass
    #         init_phi = np.array(np.random.normal(0, 0.1,
    #                             (self.number_of_layers.value, extended_image_height, extended_image_width)),ndmin=3)
    #     return init_phi

    def split_objects(self, labels_in, radius):
        #TODO implement it
        pass

    @staticmethod
    def centers_of_ml_labels(labels):
        unique_labels = np.unique(labels)
        if not len(unique_labels):
            return []
        elif len(unique_labels) == 1 and not unique_labels[0]:
            return []
        else:
            if labels.ndim > 2:
                centers = np.zeros((len(unique_labels) - 1, 2))
                maxind = 0
                for i in range(labels.shape[0]):
                    layer = labels[i, :, :]
                    layer_unique_values = np.unique(layer)
                    if not len(layer_unique_values) or len(layer_unique_values) == 1 and not layer_unique_values[0]:
                        pass
                    else:
                        cs = np.array(scipy.ndimage.center_of_mass(layer > 0, layer, layer_unique_values[1:]))
                        centers[maxind:maxind + len(layer_unique_values) - 1, :] = cs
                        maxind += len(layer_unique_values) - 1

                return centers
            else:
                return np.array(scipy.ndimage.center_of_mass(labels > 0, labels, unique_labels[1:]))

    def sort_grayscale_objects_to_layers_coloring(self, labels_in, number_of_colors):
        init_phi = np.zeros((number_of_colors,labels_in.shape[0],labels_in.shape[1]))
        colors = IdentifyPrimaryObjectsMLGOC.patch_color_select(labels_in, number_of_colors)

        colors = np.array(colors)

        for ll in range(self.number_of_layers.value):
            object_indices_of_layer = np.where(colors==ll)[0]
            temp_layer = np.zeros(labels_in.shape)-1
            for oi in object_indices_of_layer:
                temp_layer[labels_in == oi+1] = 1
            init_phi[ll, :, :] = temp_layer
        return init_phi

    @staticmethod
    def patch_color_select(labels_in, number_of_colors):
        centers = np.array(scipy.ndimage.center_of_mass(labels_in, labels_in, np.unique(labels_in)[1:]))
        ic = np.argsort(map(lambda x: x[0] * x[0] + x[1] * x[1], centers))
        n = len(centers)
        ls = [i % number_of_colors for i in range(n)]
        colors = [ls[ic[i]] for i in range(n)]
        for k1 in range(number_of_colors, n):
            k0 = ic[:k1]
            k = ic[k1]
            cd = map(lambda xx: (xx[0] - centers[k][0]) * (xx[0] - centers[k][0]) + (xx[1] - centers[k][1]) * (xx[1] - centers[k][1]),
                     centers[k0])
            s = 0
            cc = [colors[i] for i in k0]
            for c in range(number_of_colors):
                ccind = [i for i, x in enumerate(cc) if x == c]
                if ccind:
                    ind = min([cd[i] for i in ccind])
                    if ind > s:
                        s = ind
                        iss = c
            colors[k] = iss
            kc = np.where(np.logical_and(np.array(cd) == s, np.array(cc) == iss))[0][0]
            kk = k0[kc]
            k0[kc] = k
            k = kk
            kd = map(lambda xx: (xx[0] - centers[k][0]) * (xx[0] - centers[k][0]) + (xx[1] - centers[k][1]) * (xx[1] - centers[k][1]),
                     centers[k0])
            ks = 0
            cc = [colors[i] for i in k0]
            for c in range(number_of_colors):
                ccind = [i for i, x in enumerate(cc) if x == c]
                if ccind:
                    ind = min([kd[i] for i in ccind])
                    if ind > ks:
                        ks = ind
                        ik = c
            if ks > s:
                colors[k] = ik

        return colors

    def filter_on_embedded(self, labels):
        discard_incides = []
        for ll in range(self.number_of_layers.value):
            uv = np.unique(labels[ll,:,:])
            if len(uv) > 1:
                for Object in uv[1:]:
                    for lll in range(self.number_of_layers.value):
                        if ll==lll:
                            continue
                        else:
                            # compare the union of an object with the same layer
                            if np.array_equal(np.logical_or(labels[ll,:,:]==Object, labels[lll,:,:]>0), labels[lll,:,:]>0):
                                discard_incides.append(Object)
        for ind in discard_incides:
            labels[labels==ind] = 0
        labels, object_count = scipy.ndimage.label(labels)
        print('{} embedded object were removed.'.format(len(discard_incides)))
        return labels.copy()

    def filter_on_size(self, labels, object_count):
        if object_count > 0:
            areas = scipy.ndimage.measurements.sum(np.ones(labels.shape),
                                                   labels,
                                                   np.array(np.unique(labels),dtype=np.int32))
            areas = np.array(areas,dtype=int)
            min_allowed_area = np.pi * (self.preferred_radius.value * self.preferred_radius.value)
            area_image = areas[labels]
            labels[area_image < min_allowed_area] = 0
            small_removed_labels = labels.copy()
        else:
            small_removed_labels = labels.copy()
        return labels, small_removed_labels

    def merge_overlapping_objects(self, labels, JI_threshold=0.7, radius=None):
        """Merge objects that overlap and their Jaccard index is over a threshold
        """
        if radius is None:
            radius = self.preferred_radius.value
        ooindices, ool, oocentroids = self.get_overlapping_object_indices(labels)
        removed_idx_list = []
        for ooi_index, ooi in enumerate(ooindices):
            if ooi not in removed_idx_list:
                for ooj_index, ooj in enumerate(ooindices):
                    if ooj not in removed_idx_list and ooi != ooj:
                        if (oocentroids[ooi_index][0]-oocentroids[ooj_index][0])**2+(oocentroids[ooi_index][1]-oocentroids[ooj_index][1])**2 < radius**2*100:
                            obj_i = labels[ool[ooi_index], :, :].copy()
                            obj_i[obj_i != ooi] = 0
                            obj_i[obj_i == ooi] = 1
                            obj_j = labels[ool[ooj_index], :, :].copy()
                            obj_j[obj_j != ooj] = 0
                            obj_j[obj_j == ooj] = 1
                            print('JI({},{})={}'.format(ooi,ooj,jaccard_index(obj_i,obj_j)))
                            if jaccard_index(obj_i, obj_j) > JI_threshold:
                                merged_object = np.logical_or(obj_i, obj_j)
                                labels[ool[ooi_index], :, :] -= ooi*(obj_i-merged_object)
                                labels[ool[ooj_index], :, :] -= ooj*obj_j
                                removed_idx_list.append(ooj)
        print('{} objects were merged.'.format(len(removed_idx_list)))
        return labels.copy()

    def filter_on_overlap_level(self, labels, JI_threshold, radius=None):
        if radius is None:
            radius = self.preferred_radius.value
        ooindices, ool, ooc = self.get_overlapping_object_indices(labels)
        removed_indices_list = []
        for ooi_index, ooi in enumerate(ooindices):
            obj_i = labels[ool[ooi_index],:,:]
            obj_i[obj_i != ooi] = 0
            obj_i[obj_i == ooi] = 1
            if ooi not in removed_indices_list:
                area_i = obj_i.sum()
                for ooj_index, ooj in enumerate(ooindices):
                    if ooj not in removed_indices_list and ooi!=ooj:
                        if (ooc[ooi_index][0]-ooc[ooj_index][0])**2+(ooc[ooi_index][1]-ooc[ooj_index][1])**2 < radius**2 * 100:
                            obj_j = labels[ool[ooj_index], :, :]
                            obj_j[obj_j!=ooj] = 1
                            obj_j[obj_j==ooj] = 0
                            area_j = obj_j.sum()

                            if area_i > area_j:
                                bigger_index = ooi
                            else:
                                bigger_index = ooj

                            if bigger_index in removed_indices_list:
                                continue

                            if area_i > area_j:
                                ji = jaccard_index(obj_j, np.logical_and(obj_i,obj_j))
                            else:
                                ji = jaccard_index(obj_i, np.logical_and(obj_i, obj_j))

                            if ji>JI_threshold:
                                removed_indices_list.append(bigger_index)

        for removed_index in removed_indices_list:
            labels[labels == removed_index] = 0

        print('{} object(s) were(was) discarded.'.format(len(removed_indices_list)))
        return labels

    def filter_on_border(self, labels):
        """Filter out objects touching the border

        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        pass

    # def filter_on_border(self, image, labeled_image):
    #     """Filter out objects touching the border
    #
    #     In addition, if the image has a mask, filter out objects
    #     touching the border of the mask.
    #     """
    #     return labeled_image
    #     """
    #     if self.exclude_border_objects.value:
    #         border_labels = list(labeled_image[0,:])
    #         border_labels.extend(labeled_image[:,0])
    #         border_labels.extend(labeled_image[labeled_image.shape[0]-1,:])
    #         border_labels.extend(labeled_image[:,labeled_image.shape[1]-1])
    #         border_labels = np.array(border_labels)
    #         #
    #         # the following histogram has a value > 0 for any object
    #         # with a border pixel
    #         #
    #         histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
    #                                              (border_labels,
    #                                               np.zeros(border_labels.shape))),
    #                                              shape=(np.max(labeled_image)+1,1)).todense()
    #         histogram = np.array(histogram).flatten()
    #         if any(histogram[1:] > 0):
    #             histogram_image = histogram[labeled_image]
    #             labeled_image[histogram_image > 0] = 0
    #         elif image.has_mask:
    #             # The assumption here is that, if nothing touches the border,
    #             # the mask is a large, elliptical mask that tells you where the
    #             # well is. That's the way the old Matlab code works and it's duplicated here
    #             #
    #             # The operation below gets the mask pixels that are on the border of the mask
    #             # The erosion turns all pixels touching an edge to zero. The not of this
    #             # is the border + formerly masked-out pixels.
    #             mask_border = np.logical_not(scipy.ndimage.binary_erosion(image.mask))
    #             mask_border = np.logical_and(mask_border,image.mask)
    #             border_labels = labeled_image[mask_border]
    #             border_labels = border_labels.flatten()
    #             histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
    #                                                  (border_labels,
    #                                                   np.zeros(border_labels.shape))),
    #                                                   shape=(np.max(labeled_image)+1,1)).todense()
    #             histogram = np.array(histogram).flatten()
    #             if any(histogram[1:] > 0):
    #                 histogram_image = histogram[labeled_image]
    #                 labeled_image[histogram_image > 0] = 0
    #     return labeled_image
    #     """
    
    def display(self, workspace, figure):
        """Display the image and labeling"""
        if self.show_window:
            figure.set_subplots((2, 2))
            
            orig_axes     = figure.subplot(0,0)
            label_axes    = figure.subplot(1,0, sharexy = orig_axes)
            outlined_axes = figure.subplot(0,1, sharexy = orig_axes)
            final_axes = figure.subplot(1,1, sharexy = orig_axes)

            title = "Input image, cycle #%d" % (workspace.measurements.image_number,)
            image = workspace.display_data.image
            ax = figure.subplot_imshow_grayscale(0, 0, image, title)

            labels_in = workspace.display_data.labels_in
            title = "Input labels, cycle#%d" % (workspace.measurements.image_number,)
            figure.subplot_imshow_labels(1, 0, labels_in,
                                         title,
                                         sharexy = ax)

            labeled_image = workspace.display_data.labeled_image
            layer_colors = [(0,255,0),(0,127,255),(0,0,255),(0,255,255),(0,255,127)]
            cplabels = [dict(name=self.object_name.value+" ({}. layer)".format(ll+1),
                        labels=[labeled_image[ll,:,:]],
                        outline_color=cpp.tuple_to_color(layer_colors[ll%self.number_of_layers.value]))
                        for ll in range(self.number_of_layers.value)]
            title = "{} outlines".format(self.object_name.value)
            figure.subplot_imshow_grayscale(
                0, 1, image, title, cplabels=cplabels, sharexy=ax)

            filtered_image = workspace.display_data.filtered_image
            cplabels = [dict(name=self.object_name.value + " ({}. layer)".format(ll+1),
                             labels=[filtered_image[ll, :, :]],
                             outline_color=cpp.tuple_to_color(layer_colors[ll % self.number_of_layers.value])) for ll in
                        range(self.number_of_layers.value)]
            title = "Filtered {} outlines".format(self.object_name.value)
            figure.subplot_imshow_grayscale(
                1, 1, image, title, cplabels=cplabels, sharexy=ax)

    def is_object_identification_module(self):
        """DetectSpots makes primary objects so it's a identification module"""
        return True
    
    def get_measurement_objects_name(self):
        """Return the name to be appended to image measurements made by module
        """
        return self.object_name.value
    
    def get_measurement_columns(self, pipeline):
        """Column definitions for measurements made by IdentifyPrimaryObjectsMLGOC"""
        columns = cpmi.get_object_measurement_columns(self.object_name.value)
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        result = self.get_object_categories(pipeline, object_name,
                                             {self.object_name.value: [] })
        return result
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = self.get_object_measurements(pipeline, object_name, category,
                                               {self.object_name.value: [] })
        return result
    
    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """Return the objects associated with image measurements
        
        """
        return self.get_threshold_measurement_objects(pipeline, object_name,
                                                      category, measurement)

    @staticmethod
    def get_overlapping_object_indices(layers):
        if layers.ndim > 2:
            layer_num = layers.shape[0]
            ltemp = layers.copy()
            ltemp[ltemp > 0] = 1
            overlap_map = np.sum(ltemp, axis=0)
            overlap_map[overlap_map > 1] = 2
        else:
            layer_num = 1
            overlap_map = layers > 0
        ooindices = []
        ool = []
        ooc = []
        for ll in range(layer_num):
            uv = np.unique(layers[ll, :, :])
            if len(uv) > 1:
                for obj in uv[1:]:
                    object_on_layer = layers[ll,:,:]
                    # object_on_layer[object_on_layer != obj] = 0
                    # object_on_layer[object_on_layer == obj] = 1
                    hit_test = np.logical_and(overlap_map == 2, scipy.ndimage.binary_erosion(object_on_layer == obj,
                                                            structure=np.array([[0,1,0],[1,1,1],[0,1,0]])) )
                    if hit_test.any():
                        ooindices.append(obj)
                        ool.append(ll)
                        centr = scipy.ndimage.center_of_mass(object_on_layer)
                        ooc.append(list(centr))
        return ooindices, ool, ooc


IdentifyPrimaryMLGOC = IdentifyPrimaryObjectsMLGOC


def centers_of_ml_labels(labels):
    unique_labels = np.unique(labels)
    if not len(unique_labels):
        return []
    elif len(unique_labels) == 1 and not unique_labels[0]:
        return []
    else:
        if labels.ndim > 2:
            centers = []
            # centers = np.zeros((len(unique_labels)-1,2))
            # maxind = 0
            for i in range(labels.shape[0]):
                layer = labels[i,:,:]
                layer_unique_values = np.unique(layer)
                if not len(layer_unique_values) or len(layer_unique_values) == 1 and not layer_unique_values[0]:
                    pass
                else:
                    cs = scipy.ndimage.center_of_mass(layer>0, layer, layer_unique_values[1:])
                    # cs = np.array( scipy.ndimage.center_of_mass(layer>0, layer, layer_unique_values[1:]) )
                    centers += cs
                    # centers[maxind:maxind+len(layer_unique_values)-1, :] = cs
                    # maxind += len(layer_unique_values)-1

            return np.array(centers)
        else:
            return np.array( scipy.ndimage.center_of_mass(labels>0,labels,unique_labels[1:]) )


def add_object_location_measurements_ml(measurements, object_name, labels, object_count=None):
    """Add the X and Y centers of mass to the measurements

    measurements - the measurements container
    object_name  - the name of the objects being measured
    labels       - the label matrix
    object_count - (optional) the object count if known, otherwise
                   takes the maximum value in the labels matrix which is
                   usually correct.
    """
    if object_count is None:
        object_count = len(np.unique(labels))-1
    #
    # Get the centers of each object - center_of_mass <- list of two-tuples.
    #
    if object_count:
        centers = centers_of_ml_labels(labels)
        # centers = scipy.ndimage.center_of_mass(np.ones(labels.shape),
        #                                        labels,
        #                                        range(1, object_count + 1))
        centers = np.array(centers)
        # centers = centers.reshape((object_count, 2))
        location_center_y = centers[:, 0]
        location_center_x = centers[:, 1]
        number = np.array(np.unique(labels)[1:])
        # number = np.arange(1, object_count + 1)
    else:
        location_center_y = np.zeros((0,), dtype=float)
        location_center_x = np.zeros((0,), dtype=float)
        number = np.zeros((0,), dtype=int)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_X,
                                 location_center_x)
    measurements.add_measurement(object_name, M_LOCATION_CENTER_Y,
                                 location_center_y)
    measurements.add_measurement(object_name, M_NUMBER_OBJECT_NUMBER, number)


def jaccard_index(o1, o2):
    im1 = np.asarray(o1).astype(np.bool)
    im2 = np.asarray(o2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())
