---
layout: page
title:
permalink: assets/project/dfc-dit
mathjax: true
---

<center> <h2><b>Deep Feature Consistent Deep Image Transformations: Downscaling, Decolorization and HDR Tone Mapping</b></h2> </center>

<center> Xianxu Hou, Jiang Duan, Guoping Qiu </center>

<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/teaser_image.png' width="100%"></center></div>


<div style="background-color:#f7f6f1">
<h4 style="padding:10px"><b>Abstract</b></h4>
<p>
Building on crucial insights into the determining factors of the visual integrity of an image and the property of deep convolutional neural network (CNN), we have developed the Deep Feature Consistent Deep Image Transformation (DFC-DIT) framework which unifies challenging one-to-many mapping image processing problems such as image downscaling, decolorization (colour to grayscale conversion) and high dynamic range (HDR) image tone mapping. We train one CNN as a non-linear mapper to transform an input image to an output image following what we term the deep feature consistency principle which is enforced through another pretrained and fixed deep CNN. This is the first work that uses deep learning to solve and unify these three common image processing tasks. We present experimental results to demonstrate the effectiveness of the DFC-DIT technique and its state of the art performances.
</p>
</div>

<a href=""><img src='/assets/dfc-dit/web_miniature.png' width="100%"></a>


<div style="background-color:#f7f6f1">
<h4 style="padding:10px"><b>Overview</b></h4>
<p>We seek to train a convolutional neural network as a non-linear mapper to transform an input image to an output image following what we call the deep feature consistent principle. Our system consists of two components: a transformation network and a loss network. A convolutional neural network transforms an input to an output and a pretrained deep CNN is used to compute feature perceptual loss for the training of the transformation network.
</p>
</div>


<div><img src='/assets/dfc-dit/overview.png' width="100%"></div>


<div style="background-color:#f7f6f1">
	<h4 style="padding:10px"><b>Results</b></h4>
	<p>We provide the comparison results for image downscaling, decolorization and HDR image tone mapping. Additional results trained with different level feature perceptual loss are also provided for each task.</p>
</div>



<li style="background-color:#f7f6f1"><b>Image Downscaling</b> <p>Click <a href="/assets/dfc-dit/image-downscaling.html">here</a> for more results</p></li>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/downscaling_other.png' width="100%"></center></div>
<p>A comparison with different level feature perceptual loss.</p>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/downscaling_dfc.png' width="100%"></center></div>

<li style="background-color:#f7f6f1"><b>Image Decolorization</b></li>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/decolorization.png' width="100%"></center></div>
<p>A comparison with different level feature perceptual loss.</p>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/decolorization_dfc.png' width="100%"></center></div>

<li style="background-color:#f7f6f1"><b>HDR Image Tone Mapping</b></li>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/hdr.png' width="100%"></center></div>
<p>A comparison with different level feature perceptual loss.</p>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/hdr_dfc.png' width="100%"></center></div>
<p>A comparison with different gamma adjustment.</p>
<div style="background-color:#f7f6f1"><center><img src='/assets/dfc-dit/hdr_gamma.png' width="100%"></center></div>

