---
layout: page
title: 
permalink: assets/project/osc-face-detection
---

<center> <h2><b>Object Specific Deep Feature and Its Application to Face Detection</b></h2> </center>

<center> Xianxu Hou, Ke Sun, Linlin Shen, Guoping Qiu </center>
<div style="background-color:#f7f6f1">
<h4 style="padding:10px"><b>Abstract</b></h4>
<p>
We present a method for discovering and exploiting object specific deep features and use face detection as a case study. Motivated by the observation that certain convolutional channels of a Convolutional Neural Network (CNN) exhibit object specific responses, we seek to discover and exploit the convolutional channels of a CNN in which neurons are activated by the presence of specific objects in the input image. A method for explicitly fine-tuning a pre-trained CNN to induce an object specific channel (OSC) and systematically identifying it for the human face object has been developed. Building on the basic OSC features, we introduce a multi-scale approach to constructing robust face heatmaps for rapidly filtering out non-face regions thus significantly improving search efficiency for face detection in unconstrained settings. We show that multi-scale OSC can be used to develop simple and compact face detectors with state of the art performance.
</p>
</div>

<a href=""><img src='/assets/osc-face-detection/web_miniature.png' width="100%"></a>


<div style="background-color:#f7f6f1">
<h4 style="padding:10px"><b>Overview</b></h4>
</div>


<div><img src='/assets/osc-face-detection/overview.png' width="100%"></div>


<div style="background-color:#f7f6f1">
	<h4 style="padding:10px"><b>Results</b></h4>
</div>

<li style="background-color:#f7f6f1"><b>Real-time face detection demo running on CPU</b></li>

<div style="background-color:#f7f6f1"><center><img src='/assets/osc-face-detection/realtime_demo.gif' width="40%"></center></div>

<li style="background-color:#f7f6f1"><b>Test on <a href="http://vis-www.cs.umass.edu/fddb/">FDDB</a> dataset</b></li>

<img src='/assets/osc-face-detection/roc_results_fddb.png' width="100%">

<li style="background-color:#f7f6f1"><b>Qualitative face detection results</b></li>

<img src='/assets/osc-face-detection/detected_examples.png' width="100%">


