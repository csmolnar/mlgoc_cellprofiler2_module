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
<li><i>Upper right:</i> The segmented objects as colored image.
It is important to note that assigned colors are 
arbitrary; they are used simply to help you distinguish the various objects. </li>
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

The module first calculates a trous wavelet (Olivo-Marin, 2002) representation of the image.
User-defined level of the wavelet is extracted and thresholded to remove noise.
Next the thresholded wavelet image is smoothed with circular average filter and Gaussian filter.
Finally, the locations of spots are detected by finding the local maxima from the image.
These spots are thresholded with user-defined static threshold.

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
from cellprofiler.cpmath.cpmorphology import relabel
import cellprofiler.cpmath.outline
import cellprofiler.objects
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP
import compute_mlgoc_parameters

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

# Settings text which is referenced in various places in the help

PREFERRED_RADIUS_TEXT = "Preferred radius in pixels"


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
            What did you call the initial seeds for correction?""")

        self.object_name = cps.ObjectNameProvider(
            "Name the primary objects to be identified",
            "CorrectedNuclei", doc="""
            Enter the name that you want to call the objects identified by this module.""")

        self.preferred_radius = cps.Integer(
            PREFERRED_RADIUS_TEXT, 20, doc="""
            Give the approximate radius of object to identify.""")

        self.number_of_layers_string = cps.Text(
            "Number of layers", 'Automatic', doc="""
            Give the number of overlapping layers. The default number is 4.""")

        self.mu_in = cps.Float(
            "Mean intensity of a single objects", 0.75, doc="""
            Give a mean intensity of a single (not overlapping) object.""")

        self.sigma_in = cps.Float(
            "Variance of intensities over single objects", 0.05, doc="""
            Give the variance of the intensities over single (not overlapping) objects.""")

        self.mu_out = cps.Float(
            "Mean intensity the background", 0.25, doc="""
            Give a mean intensity the image background.""")

        self.sigma_out = cps.Float(
            "Variance of intensities over background", 0.05, doc="""
            Give the variance of the intensities over the background.""")

        self.data_weight = cps.Float(
            "Weight of data term", 1.0, doc="""
            Give the weight of data term compared to the shape energy.""")

        self.initialization_type = cps.Choice(
            "Type of initialization", ["Seeds (manual)", "Circular seeds (manual)", "Neutral", "Squares"], doc="""
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
                self.preferred_radius, self.number_of_layers_string,
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
                self.preferred_radius, self.number_of_layers_string,
                self.mu_in, self.sigma_in, self.mu_out, self.sigma_out,
                self.data_weight, self.initialization_type,
                self.maximum_iterations, self.exclude_border_objects,
                self.overlap_ji_threshold,
                self.should_save_outlines, self.save_outlines]

    def visible_settings(self):
        vv = [self.image_name, self.seed_objects, self.object_name,
                self.preferred_radius, self.number_of_layers_string,
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

        image_width = image.shape[0]
        image_height = image.shape[1]

        alpha_tilde = 1.0
        lambda_tilde = 1.0
        rhatstar = 1.0
        kappa = 0.0

        p = compute_mlgoc_parameters.compute_mlgoc_parameters(alpha_tilde,
                                                              lambda_tilde,
                                                              self.preferred_radius.value,
                                                              rhatstar)

        prior_phase_field_parameters = [p[1] for i in range(3)]

        objects = workspace.object_set.get_objects(self.seed_objects.value)

        gradient_weight = 0.0

        data_parameters = {'muin': self.mu_in,
                           'muout': self.mu_out,
                           'sigmain': self.sigma_in,
                           'sigmaout': self.sigma_out,
                           'gamma1': gradient_weight,
                           'gamma2': self.data_weight}

        maxd = int(max(map(lambda x: x['d'], prior_phase_field_parameters)) )

        extended_image = np.pad(image, ((2*maxd,2*maxd),(2*maxd,2*maxd)), 'constant', constant_values=(self.mu_out.value,))
        extended_image_height = extended_image.shape[0]
        extended_image_width = extended_image.shape[1]

        # distribution of objects into several layers

        optimization_parameters = {'tolerance': 1e-10,
                                   'max_iterations': self.maximum_iterations.value,
                                   'save_frequency': -1}

        # initial_phi = self.initialize_phi()

        if self.show_window:
            workspace.display_data.image = extended_image
            workspace.display_data.mask = mask


        # workspace.display_data.statistics = []
        # level = int(self.atrous_level.value)
        #
        # wavelet = self.a_trous(1.0*image, level+1)
        # wlevprod = wavelet[:,:,level-1] * 3.0
        #
        # spotthresh = wlevprod.mean() + float(self.noise_removal_factor.value) * wlevprod.std()
        # tidx = wlevprod < spotthresh
        # wlevprod[tidx] = 0
        #
        # wlevprod = self.circular_average_filter(wlevprod, int(self.smoothing_filter_size.value))
        # wlevprod = self.smooth_image(wlevprod, mask)
        #
        # max_wlevprod = scipy.ndimage.filters.maximum_filter(wlevprod,3)
        # maxloc = (wlevprod == max_wlevprod)
        # twlevprod = max_wlevprod > float(self.final_spot_threshold.value)
        # maxloc[twlevprod == 0] = 0
        #
        # labeled_image,object_count = scipy.ndimage.label(maxloc,
        #                                                  np.ones((3,3),bool))
        #
        # unedited_labels = labeled_image.copy()
        # # Filter out objects touching the border or mask
        # border_excluded_labeled_image = labeled_image.copy()
        # labeled_image = self.filter_on_border(image, labeled_image)
        # border_excluded_labeled_image[labeled_image > 0] = 0
        #
        # # Relabel the image
        # labeled_image,object_count = relabel(labeled_image)
        # new_labeled_image, new_object_count = self.limit_object_count(
        #     labeled_image, object_count)
        # if new_object_count < object_count:
        #     # Add the labels that were filtered out into the border
        #     # image.
        #     border_excluded_mask = ((border_excluded_labeled_image > 0) |
        #                             ((labeled_image > 0) &
        #                              (new_labeled_image == 0)))
        #     border_excluded_labeled_image = scipy.ndimage.label(border_excluded_mask,
        #                                                         np.ones((3,3),bool))[0]
        #     object_count = new_object_count
        #     labeled_image = new_labeled_image
        #
        # # Make an outline image
        # outline_image = cellprofiler.cpmath.outline.outline(labeled_image)
        # outline_border_excluded_image = cellprofiler.cpmath.outline.outline(border_excluded_labeled_image)
        #
        # if self.show_window:
        #     statistics = workspace.display_data.statistics
        #     statistics.append(["# of accepted objects",
        #                        "%d"%(object_count)])
        #
        #     workspace.display_data.image = image
        #     workspace.display_data.labeled_image = labeled_image
        #     workspace.display_data.border_excluded_labels = border_excluded_labeled_image
        #
        # # Add image measurements
        # objname = self.object_name.value
        # measurements = workspace.measurements
        # cpmi.add_object_count_measurements(measurements,
        #                                    objname, object_count)
        # # Add label matrices to the object set
        # objects = cellprofiler.objects.Objects()
        # objects.segmented = labeled_image
        # objects.unedited_segmented = unedited_labels
        # objects.parent_image = image
        #
        # workspace.object_set.add_objects(objects,self.object_name.value)
        # cpmi.add_object_location_measurements(workspace.measurements,
        #                                       self.object_name.value,
        #                                       labeled_image)
        # if self.should_save_outlines.value:
        #     out_img = cpi.Image(outline_image.astype(bool),
        #                         parent_image = image)
        #     workspace.image_set.add(self.save_outlines.value, out_img)

    def initialize_phi(self, initial_phi):
        pass

    def a_trous(self, image, level):
        '''
        Calculates a trous wavelet in image up to the level
        '''
        wavelet = np.zeros((image.shape[0],image.shape[1],level))
        cvol = np.zeros((image.shape[0],image.shape[1],level))
        cvol[:,:,0] = image
        
        for i in xrange(1,level):
            ccimg = self.calculateC(image, i-1)
            cvol[:,:,i] = ccimg
            wavelet[:,:,i-1] = cvol[:,:,i-1] - cvol[:,:,i]
        wavelet[:,:,level-1] = cvol[:,:,level-1]

        return wavelet

    def calculateC(self, img, level):
        '''
        '''
        rlevel = 2**level
        if img.shape[0] < 2*rlevel or img.shape[1] < 2*rlevel:
            return img

        pimg = np.pad(img, 2*rlevel, 'reflect')
        
        def conv_filter(sigma, size, gap = 0):
            filt = np.zeros(((size-1)*(gap+1)+1, (size-1)*(gap+1)+1),dtype='float')
            middle = size / 2
            for i in xrange(size):
                for j in xrange(size):
                    dist = np.sqrt((middle-i+0.5)**2 + (middle-j+0.5)**2)
                    normpdf = scipy.stats.norm.pdf(dist/sigma) / sigma
                    filt[(i-1)*(gap+1)+1, (j-1)*(gap+1)+1] = normpdf
            filt = filt / filt.sum()
            return filt

        filt = conv_filter(1, 5, level**2)
        cimg = scipy.ndimage.filters.convolve(pimg,filt)
        cimg = cimg[2*rlevel:img.shape[0]+2*rlevel, 2*rlevel:img.shape[1]+2*rlevel]
        return cimg

    def circular_average_filter(self, image, radius):
        '''Blur an image using a circular averaging filter (pillbox)  
        image - grayscale 2-d image
        radii - radius of filter in pixels

        The filter will be within a square matrix of side 2*radius+1

        This code is translated directly from MATLAB's fspecial function
        '''
        crad = np.ceil(radius-0.5)
        x,y = np.mgrid[-crad:crad+1,-crad:crad+1].astype(float) 
        maxxy = np.maximum(abs(x),abs(y))
        minxy = np.minimum(abs(x),abs(y))

        m1 = ((radius **2 < (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) + 
          (radius**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2) * 
          np.real(np.sqrt(np.asarray(radius**2 - (maxxy + 0.5)**2,dtype=complex)))) 
        m2 = ((radius**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) + 
          (radius**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)*
          np.real(np.sqrt(np.asarray(radius**2 - (maxxy - 0.5)**2,dtype=complex))))

        sgrid = ((radius**2*(0.5*(np.arcsin(m2/radius) - np.arcsin(m1/radius)) + 
              0.25*(np.sin(2*np.arcsin(m2/radius)) - np.sin(2*np.arcsin(m1/radius)))) - 
             (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5)) *  
             ((((radius**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & 
             (radius**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | 
             ((minxy == 0) & (maxxy-0.5 < radius) & (maxxy+0.5 >= radius)))) ) 

        sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < radius**2) 
        sgrid[crad,crad] = np.minimum(np.pi*radius**2,np.pi/2) 
        if ((crad>0) and (radius > crad-0.5) and (radius**2 < (crad-0.5)**2+0.25)): 
            m1  = np.sqrt(radius**2 - (crad - 0.5)**2) 
            m1n = m1/radius 
            sg0 = 2*(radius**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
            sgrid[2*crad,crad]   = sg0
            sgrid[crad,2*crad]   = sg0
            sgrid[crad,0]        = sg0 
            sgrid[0,crad]        = sg0
            sgrid[2*crad-1,crad] = sgrid[2*crad-1,crad] - sg0
            sgrid[crad,2*crad-1] = sgrid[crad,2*crad-1] - sg0
            sgrid[crad,1]        = sgrid[crad,1]        - sg0 
            sgrid[1,crad]        = sgrid[1,crad]        - sg0 

        sgrid[crad,crad] = np.minimum(sgrid[crad,crad],1) 
        kernel = sgrid/sgrid.sum()
        output = scipy.ndimage.filters.convolve(image, kernel, mode='constant')

        return output

    def smooth_image(self, image, mask):
        """Apply the smoothing filter to the image"""
        
        filter_size = self.smoothing_filter_size.value
        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0),1)
        f = (1/np.sqrt(2.0 * np.pi ) / sigma * 
             np.exp(-0.5 * np.arange(-filter_size, filter_size+1)**2 / 
                    sigma ** 2))
        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f,
                                              axis = 0,
                                              mode='constant')
            return scipy.ndimage.convolve1d(output, f,
                                            axis = 1,
                                            mode='constant')
        #
        # Use the trick where you similarly convolve an array of ones to find 
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image

    def limit_object_count(self, labeled_image, object_count):
        '''Limit the object count according to the rules
        
        labeled_image - image to be limited
        object_count - check to see if this exceeds the maximum
        
        returns a new labeled_image and object count
        '''
        if object_count > self.maximum_object_count.value:
            if self.limit_choice == LIMIT_ERASE:
                labeled_image = np.zeros(labeled_image.shape, int)
                object_count = 0
            elif self.limit_choice == LIMIT_TRUNCATE:
                #
                # Pick arbitrary objects, doing so in a repeatable,
                # but pseudorandom manner.
                #
                r = np.random.RandomState()
                r.seed(abs(np.sum(labeled_image)))
                #
                # Pick an arbitrary ordering of the label numbers
                #
                index = r.permutation(object_count) + 1
                #
                # Pick only maximum_object_count of them
                #
                index = index[:self.maximum_object_count.value]
                #
                # Make a vector that maps old object numbers to new
                #
                mapping = np.zeros(object_count+1, int)
                mapping[index] = np.arange(1,len(index)+1)
                #
                # Relabel
                #
                labeled_image = mapping[labeled_image]
                object_count = len(index)
        return labeled_image, object_count
        
    def filter_on_border(self, image, labeled_image):
        """Filter out objects touching the border
        
        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        return labeled_image
        """
        if self.exclude_border_objects.value:
            border_labels = list(labeled_image[0,:])
            border_labels.extend(labeled_image[:,0])
            border_labels.extend(labeled_image[labeled_image.shape[0]-1,:])
            border_labels.extend(labeled_image[:,labeled_image.shape[1]-1])
            border_labels = np.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                                 (border_labels,
                                                  np.zeros(border_labels.shape))),
                                                 shape=(np.max(labeled_image)+1,1)).todense()
            histogram = np.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = np.logical_not(scipy.ndimage.binary_erosion(image.mask))
                mask_border = np.logical_and(mask_border,image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                                     (border_labels,
                                                      np.zeros(border_labels.shape))),
                                                      shape=(np.max(labeled_image)+1,1)).todense()
                histogram = np.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image
        """
    
    def display(self, workspace, figure):
        """Display the image and labeling"""
        if self.show_window:
            figure.set_subplots((2, 2))
            
            orig_axes     = figure.subplot(0,0)
            # label_axes    = figure.subplot(1,0, sharexy = orig_axes)
            # outlined_axes = figure.subplot(0,1, sharexy = orig_axes)

            title = "Input image, cycle #%d"%(workspace.measurements.image_number,)
            image = workspace.display_data.image
            # labeled_image = workspace.display_data.labeled_image
            # border_excluded_labeled_image = workspace.display_data.border_excluded_labels

            ax = figure.subplot_imshow_grayscale(0, 0, image, title)
            # figure.subplot_imshow_labels(1, 0, labeled_image,
            #                              self.object_name.value,
            #                              sharexy = ax)
    
            # cplabels = [
            #     dict(name = self.object_name.value,
            #          labels = [labeled_image]),
            #     dict(name = "Objects touching border",
            #          labels = [border_excluded_labeled_image])]
            # title = "%s outlines"%(self.object_name.value)
            # figure.subplot_imshow_grayscale(
            #     0, 1, image, title, cplabels = cplabels, sharexy = ax)
            
            # figure.subplot_table(
            #     1, 1,
            #     [[x[1]] for x in workspace.display_data.statistics],
            #     row_labels = [x[0] for x in workspace.display_data.statistics])
    
    def is_object_identification_module(self):
        """DetectSpots makes primary objects so it's a identification module"""
        return True
    
    def get_measurement_objects_name(self):
        """Return the name to be appended to image measurements made by module
        """
        return self.object_name.value
    
    def get_measurement_columns(self, pipeline):
        """Column definitions for measurements made by DetectSpots"""
        columns = cpmi.get_object_measurement_columns(self.object_name.value)
        return columns

    def get_categories(self,pipeline, object_name):
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
