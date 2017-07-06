import cellprofiler.objects
import numpy as np


class ObjectsML(cellprofiler.objects.Objects):
    """Represents a multi-layered segmentation of an image."""
    def __init__(self):
        self.__segmented = None
        self.__unedited_segmented = None
        self.__small_removed_segmented = None
        self.__parent_image = None

    def get_segmented(self):
        return self.__segmented

    def set_segmented(self,labels):
        self.__segmented = SegmentationML(labels=labels)
        self.__small_removed_segmented = SegmentationML(labels=labels)

    segmented = property(get_segmented, set_segmented)

    def get_unedited_segmented(self):
        return self.__unedited_segmented

    def set_unedited_segmented(self,labels):
        self.__unedited_segmented = SegmentationML(labels=labels)

    unedited_segmented = property(get_unedited_segmented, set_unedited_segmented)

    @property
    def shape(self):
        return self.__segmented.get_shape()

    def get_parent_image(self):
        """The image that was analyzed to yield the objects.

        The image is an instance of CPImage which means it has the mask
        and crop mask.
        """
        return self.__parent_image

    def set_parent_image(self, parent_image):
        self.__parent_image = parent_image

    parent_image = property(get_parent_image, set_parent_image)

    def get_has_parent_image(self):
        """True if the objects were derived from a parent image

        """
        return self.__parent_image is not None

    has_parent_image = property(get_has_parent_image)

    def relate_children(self, children):
        pass

    def relate_labels(self, parent_labels, child_labels):
        pass

    @property
    def count(self):
        return len(np.unique(self.__segmented))-1

    def cache(self, hdf5_object_set, objects_name):
        pass

    def get_labels(self):
        return self.__segmented.get_labels()

    def get_unique_values(self):
        return np.unique(self.__segmented.get_labels())


class SegmentationML(object):
    def __init__(self, labels=None, shape=None):
        self.__labels = labels
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = labels.shape
            self.__explicit_shape = False

    def get_shape(self):
        if self.__shape is not None:
            return self.__shape

    def set_shape(self,shape):
        self.__shape = shape
        self.__explicit_shape = True

    def has_shape(self):
        return self.__explicit_shape

    shape = property(get_shape, set_shape)

    def get_labels(self):
        return self.__labels

    def set_labels(self,labels):
        self.__labels = labels
        self.__shape = labels.shape
