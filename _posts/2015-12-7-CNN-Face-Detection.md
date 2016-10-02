---
layout: post
title: CNN Face Dectection
excerpt: "My PhD work about object specific deep features for Face Dectection"
modified: 2015-12-7
comments: true
mathjax: true
---

The post records some notes for CNN Face Dectection project in my PhD in the University of Nottingham.


### Note 1: Make image square and crop/split it into sub_images
**Make image square**

In order to use Convolutional Neural Network that (mostly) requires the input image square, i.e. of shape (3, N, N), I need to make the height equals to width. There are 3 ways coming into my mind:

- Stretch the image to square, **not good** because the face could be stretched. 
{% highlight python %}
import cv2
import numpy as np
def stretch_to_square(frame, size):
    return cv2.resize(frame, size, size)
{% endhighlight %}

- Crop the image to square, usually set the cropped size as the smaller one of width and height. **Not good** because information could be lost
{% highlight python %}
def crop_to_square(frame):
    y_size, x_size = frame.shape[0], frame.shape[1]
    if x_size y_size:
        # landscape
        offset = (x_size - y_size) / 2
        return frame[:,offset:offset + y_size,:]
    else:
        # portrait
        offset = (y_size - x_size) / 2
        return frame[offset:offset + x_size,:,:]
{% endhighlight %}

- Padded the image with zeros to square, **better solution**. However we need to store the padded size in order to convert from padded image coordinates to original image coordinates. We can define a new class SuperImage to achieve this.
{% highlight python %}
def padding_to_square(frame, bookkeeping=False, up_scale=False):
    y_size, x_size = frame.shape[0], frame.shape[1]
    if y_size == x_size:
        padded_frame = frame
        super_image = SuperImage(padded_frame, 0, 0)
    if y_size x_size:
        pad_before = (y_size - x_size) // 2
        pad_after = y_size - x_size - pad_before
        padded_frame = np.pad(frame, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
        pad = 0
        if up_scale:  # up scale the image, more padding if needed
            pad = y_size // up_scale
            padded_frame = np.pad(padded_frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        super_image = SuperImage(padded_frame, 0 + pad, pad_before + pad)
    if y_size < x_size:
        pad_before = (x_size - y_size) // 2
        pad_after = x_size - y_size - pad_before
        padded_frame = np.pad(frame, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        pad = 0
        if up_scale:
            pad = x_size // up_scale
            padded_frame = np.pad(padded_frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        super_image = SuperImage(padded_frame, pad_before + pad, 0 + pad)
    if bookkeeping:
        return super_image
    else:
        return padded_frame

class SuperImage(object):
    """ Store the current position of a super_image (after padding) """
    def __init__(self, image, y, x):
        """
        - image: an array, represent the sub image
        - y: int to represent the top coordinate in the parent image
        - x: int to represent the left coordinate in the parent image
        """
        self.type = 'SuperImage'
        self.data = image
        self.sup_y0 = y
        self.sup_x0 = x

    def old_x_y(self, y, x):
        """ convert (y, x) in the super image to the parent / old image coordinate system """
        return (y - self.sup_y0, x - self.sup_x0)

{% endhighlight %}

**Split image into sub_images**

For simplicity (because simple is good), I use sliding window to split images. I design to overlap the sub_images to make sure any two continuous pixels can appear at least one sub_image. Specifically, if I want to get n*n sub_images, then set `stride = int(1.0/n * height)`, and sub_image size `sub_size = 2 * stride` to overlap half of the sub_images. If the size of remains are not enough to form a sub_image, we can add zero_padding or just throw them away.
In addition, we can use SubImage Class to store the information to convert sub_image coordinates back to original image.

{% highlight python %}
def split_image(image, stride_ratio=1.0/3, pad_to_fit=False):
    """ split to n*n SubImages based on window sliding """
    # make the image square
    sub_images = []
    square_image = padding_to_square(image)
    height, width, _ = square_image.shape
    stride = int(stride_ratio * height)
    sub_size = 2 * stride
    padding_image = square_image
    if pad_to_fit:
        # decide how much padding need to make sure the sliding "fit" across input neatly
        # it is like the zero_padding in the convolutional layer
        n = height // stride
        remain_size = height - n * stride
        pad_before, pad_after = 0, 0
        if remain_size = 0: # need zero_padding, note: here should include 0 to make sure every image has the same number of sub images
            pad_before = (stride - remain_size) // 2  # may not be even
            pad_after = stride - remain_size - pad_before
            padding_image = np.pad(square_image, ((pad_before, pad_after), (pad_before, pad_after), (0, 0)), mode='constant')
        else:
            padding_image = square_image
    else:
        pad_before, pad_after = 0, 0
    print 'pad', pad_before, pad_after
    pad_image_size = padding_image.shape[0]
    for y in xrange(0, pad_image_size-stride, stride): 
        for x in xrange(0, pad_image_size-stride, stride):
            if (y + sub_size) <= pad_image_size and (x + sub_size) <= pad_image_size:  # not pad to fit 
                sub_image = padding_image[y:y+sub_size, x:x+sub_size, :].copy()
                # we need the original not the padded image in above when pad_to_fit
                sub_image = SubImage(sub_image, y - pad_before, x - pad_before)  # it is a little ugly, but it works
                sub_images.append(sub_image)
    return sub_images

class SubImage(object):
    """ Store the current position of a sub_image """
    def __init__(self, image, y, x):
        """
        - image: an array, represent the sub image
        - y: int to represent the top coordinate in the parent image
        - x: int to represent the left coordinate in the parent image
        """
        self.type = 'SubImage'
        self.data = image
        self.sup_y0 = y
        self.sup_x0 = x

    def old_x_y(self, y, x):
        """ convert (y, x) in the subimage to the parent image coordinate system """
        return (y + self.sup_y0, x + self.sup_x0)

{% endhighlight %}

** Convert from subimage to original image

The process is a little tricky. Because the subimages are actually cropped from padded image, the coordinates of subimages in the coordinate of original image can be less than 0, and also bigger than the original image size (see following figure). So we should make sure the details right when convert subimage information (e.g. human face bounding box) to original image coordinates.
![padded and subimage]({{ site.url }}/images/CNNFace/1.png "padded and subimage")

Following code illustrate how to convert a subimage heatmap to the corresponding position in the coordinates of the original image

{% highlight python %}
def pad_subheatmap_to_old_size(sub_heatmap_image, sub_image, padded_square):
    """ Zero-pad sub_heatmap to the size of the old image size of which the sub_images cropped from """
    padded_heatmap = np.zeros_like(padded_square.data)
    sub_image_size = sub_heatmap_image.shape[0]
    padded_square_size = padded_square.data.shape[0]
    # get coordinate in old/super image of the origin point in the heatmap 
    y_start_old, x_start_old = sub_image.old_x_y(0, 0)
    y_end_old, x_end_old = y_start_old + sub_image_size, x_start_old + sub_image_size
    y_start_sub, x_start_sub, y_end_sub, x_end_sub = 0, 0, sub_image_size, sub_image_size

    # Here is a little tricky, please refer to image_misc.split_image
    # Note, there are two images. One is the cropped sub image and the old image which the subimage cropped from
    if y_start_old < 0:
        y_start_sub = 0 - y_start_old
        y_start_old = 0

    if x_start_old < 0:
        x_start_sub = 0 - x_start_old
        x_start_old = 0

    if y_end_old padded_square_size:
        y_end_sub = sub_image_size - (y_end_old - padded_square_size)
        y_end_old = padded_square_size

    if x_end_old padded_square_size:
        x_end_sub = sub_image_size - (x_end_old - padded_square_size)
        x_end_old = padded_square_size

    padded_heatmap[y_start_old:y_end_old, x_start_old:x_end_old, :] = sub_heatmap_image[y_start_sub:y_end_sub, x_start_sub:x_end_sub, :]
    return padded_heatmap
{% endhighlight %}


