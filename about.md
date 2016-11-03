---
layout: page
title: 'About'
permalink: /about/
---
<head> 
  <style>
    .row hr {
      margin-top: 10px;
      margin-bottom: 10px;
    }

    .wrap {
      max-width: 1000px;
      padding: 0 30px;
      margin: 0 auto;
      zoom: 1;
    }
  </style>
</head>

<div>
  <div id="timeline">
    <div class="timelineitem">
      <div><img class = "img-circle imgme" src="/images/avatar-phd2.jpg" width="120px" style="float:left"></div>     
      <div class="tdate">2015 -</div>
      <div class="ttitle">&nbsp;University of Nottingham Ningbo China</div>
      <div class="tdesc">&nbsp;School of Computer Science, Ph.D. student</div>
      <div class="tdesc">&nbsp;Computer Vision, Deep Learning</div>
      <div class="tdesc">&nbsp;Supervisor: <a href="http://www.cs.nott.ac.uk/~pszqiu/">Guoping Qiu</a></div>
    </div>
    <div class="timelineitem">
      <div><img class = "img-circle imgme" src="/images/avatar-phd1.jpg" width="120px" style="float:right"></div>     
      <div class="tdate">2014 - 2015</div>
      <div class="ttitle">University of Nottingham, Ph.D. student&nbsp;</div>
      <div class="tdesc"><a href="https://www.horizon.ac.uk">Horizon Center for Doctoral Training&nbsp;</a></div>
      <div class="tdesc">&nbsp;</div>
    </div>
    <div class="timelineitem">
      <div><img class = "img-circle imgme" src="/images/avatar-intern.jpg" width="120px" style="float:left"></div>     
      <div class="tdate">Summer 2014</div>
      <div class="ttitle">&nbsp;Weiboyi, Beijing, Internship</div>
      <div class="tdesc">&nbsp;Data Mining Group</div>
      <div class="tdesc">&nbsp;</div>
    </div>
    <div class="timelineitem">
      <div><img class = "img-circle imgme" src="/images/avatar-master.jpg" width="120px" style="float:right"></div>
     <div class="tdate">2011-2014</div>
      <div class="ttitle">China University of Mining &amp; Technology Beijing</div>
      <div class="tdesc">Master's Degree, Major in (Coal) Geology&nbsp;</div>
      <div class="tdesc">Supervisor: Yuegang Tang</div>
    </div>
    <div class="timelineitem">
      <div><img class = "img-circle imgme" src="/images/avatar-bachelor.jpg" width="120px" style="float:left"></div>      
      <div class="tdate">2007-2011</div>
      <div class="ttitle">China University of Mining &amp; Technology Beijing</div>
      <div class="tdesc">&nbsp;Bachelor's Degree, Major in Geography</div>
      <div class="tdesc"> &nbsp;</div>
    </div>
  </div>
</div>

I am **Xianxu Hou (侯贤旭)**, a PhD student at University of Nottingham working on Deep Learning and It’s application in Computer Vision.


<hr class="bold" />

<center><h2>Publications</h2></center>

<div>
  <div>
    <div class="row">
      <div class="col-md-6">
        <div class="pubimg">
          <img src="/assets/dfcvae/overview.png" width="100%">
        </div>
        <hr />
        <div class="pubimg">
          <img src="/assets/dfcvae/combined.gif" width="50%" style="float:left">
        </div>
        <div class="pubimg">
          <img src="/assets/dfcvae/random.gif" width="40%" style="float:right">
        </div>
      </div>
      <div class="col-md-6">
        <div class="pub">
          <div class="pubt">Deep Feature Consistent Variational Autoencoder</div>
          <div class="pubd">We present a novel method for constructing Variational Autoencoder (VAE). Instead of using pixel-by-pixel loss, we enforce deep feature consistency between the input and the output of a VAE, which ensures the VAE's output to preserve the spatial correlation characteristics of the input, thus leading the output to have a more natural visual appearance and better perceptual quality. Based on recent deep learning works such as style transfer, we employ a pre-trained deep convolutional neural network (CNN) and use its hidden features to define a feature perceptual loss for VAE training. Evaluated on the CelebA face dataset, we show that our model produces better results than other methods in the literature. We also show that our method can produce latent vectors that can capture the semantic information of face expressions and can be used to achieve state-of-the-art performance in facial attribute prediction.</div>
          <div class="puba"><b>Xianxu Hou</b>, Linlin Shen, Ke Sun, Guoping Qiu</div>
          <div class="publ">
            <ul>
              <li><a href="/assets/project/dfcvae">[Project]</a></li>
              <li><a href="https://arxiv.org/abs/1610.00291">[arXiv]</a></li>
              <li><a href="https://github.com/houxianxu/DFC-VAE">[Code]</a></li>
            </ul>
          </div>
        </div>
      </div>
    </div>

  <hr />
  <!-- another publication -->
  <div class="row">
    <div class="col-md-6">
      <div class="pubimg">
        <img src="/assets/osc-face-detection/overview.png" width="90%">
      </div>
      <hr />
      <div class="pubimg">
        <img src="/assets/osc-face-detection/detected_examples.png" width="90%">
      </div>
      
    </div>
    <div class="col-md-6">
      <div class="pub">
        <div class="pubt">Object Specific Deep Feature and Its Application to Face Detection</div>
        <div class="pubd">We present a method for discovering and exploiting object specific deep features and use face detection as a case study. Motivated by the observation that certain convolutional channels of a Convolutional Neural Network (CNN) exhibit object specific responses, we seek to discover and exploit the convolutional channels of a CNN in which neurons are activated by the presence of specific objects in the input image. A method for explicitly fine-tuning a pre-trained CNN to induce an object specific channel (OSC) and systematically identifying it for the human face object has been developed. Building on the basic OSC features, we introduce a multi-scale approach to constructing robust face heatmaps for rapidly filtering out non-face regions thus significantly improving search efficiency for face detection in unconstrained settings. We show that multi-scale OSC can be used to develop simple and compact face detectors with state of the art performance.</div>
        <div class="puba"><b>Xianxu Hou</b>, Ke Sun, Linlin Shen, Guoping Qiu</div>
        <div class="publ">
          <ul>
            <li><a href="/assets/project/osc-face-detection">[Project]</a></li>
            <li><a href="https://arxiv.org/abs/1609.01366">[arXiv]</a></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <hr />
  <!-- another publication -->
  <div class="row">
    <div class="col-md-6">
      <div class="pubimg">
        <img src="/assets/ke-avtd/parse.png" width="90%">
      </div>
      <hr />
      <div class="pubimg">
        <img src="/assets/ke-avtd/retrieval.png" width="90%">
      </div>  
    </div>
    <div class="col-md-6">
      <div class="pub">
        <div class="pubt">Automatic Visual Theme Discovery from Joint Image and Text Corpora</div>
        <div class="pubd">We propose an unsupervised visual theme discovery framework as a better alternative for semantic representation of visual contents. We first show that tag based annotation lacks consistency and compactness for describing visually similar contents. We then learn the visual similarity between tags based on the visual features of the images containing the tags. At the same time, we use a natural language processing technique to measure the semantic similarity between tags. Finally, we cluster tags into visual themes based on their visual similarity and semantic similarity measures using a spectral clustering algorithm. We then design three common computer vision tasks, example based image search, keyword based image search and image labeling to explore potential application of our visual themes discovery framework. In experiments, visual themes significantly outperform tags on semantic image understanding and achieve state-of-art performance in all three tasks.</div>
        <div class="puba">Ke Sun, <b>Xianxu Hou</b>, Qian Zhang, Guoping Qiu</div>
        <div class="publ">
          <ul>
            <li><a href="https://arxiv.org/abs/1609.01859">[arXiv]</a></li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <hr>

  <!-- Geology Publications  -->
  <h4>Some publications in (Coal) Geology</h4>

  <div class="row">
    <div class="col-md-6">
      <div class="pubimg">
        <img src="/assets/geology/guizhou.png">
      </div>
 
    </div>
    <div class="col-md-6">
      <div class="pub">
        <div class="pubt">Geochemistry of Tuffs and Tonstein from the Mayi Exploration Area, Guizhou China</div>
        <div class="pubd">Mineralogical and geochemical characteristics of the Late Permian tuffs and tonsteins are studied by X-ray fluorescence(XRF),powder X-ray diffraction(XRD),and inductively-coupled plasma mas spectrometry (ICP-MS).The results show that clay minerals in most tuffs and tonsteins are dominated by kaolinite. All the samples are rich in trace elements Nb, Ta, Zr, Hf and REEs, and contains high content of minerals like pyrite due to the seawater influences. The primitive-mantle normalized spider-diagrams are similar to the ocean island basalt (OIB), which indicates that the volcanic ash forming tuffs and tonsteins may originate from a waning mantle plume, furthermore the ratio of Nb/Ta suggests a contamination from continental crust or lithosphere.</div>
        <div class="puba"><b>Xianxu Hou</b>, Yuegang Tang, Cong Wang, Weicheng Gao, Qiang Wei</div>
        <div class="pubv">Bulletin of Mineralogy, Petrology and Geochemistry, 2014</div>
        <div class="publ">
          <ul>
            <li><a href="/assets/geology/guizhou.pdf">[pdf]</a>&nbsp;(Chinese with English abstract)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <hr />

<!-- net publication -->
  <div class="row">
    <div class="col-md-6">
      <div class="pubimg">
        <img src="/assets/geology/chongqing.png">
      </div>
 
    </div>
    <div class="col-md-6">
      <div class="pub">
        <div class="pubt">Coal Petrology and Coal facies of Zhongliangshan Mining Area, Chongqing China</div>
        <div class="pubd">The primary coal seams in Zhongliangshan mining area are analyzed by macerals ternary diagrams and coal facies diagrams based on the maceral classification of ICCP 1994 system. The coal seams are characterized by the predominance of semi-bright and structural coals, which is a result of Epigenetic tectonic activity. The petrographic composition is dominated by vitrinite, and collodetrinite and vitrodetrinite are the major macerals. The sedimentary environments is deduced from petrographic composition and coal facies diagrams demonstrate the transition from low delta plain to limnic environment, and the mires were influenced by marine flooding.</div>
        <div class="puba"><b>Xianxu Hou</b>, Yuegang Tang, Xiaoxia Song, Mingxian Yang, Mingtao Guo, Long Jia</div>
        <div class="pubv">Coal Geology &amp; Exploration, 2013</div>
        <div class="publ">
          <ul>
            <li><a href="/assets/geology/chongqing.pdf">[pdf]</a>&nbsp;(Chinese with English abstract)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <hr />

<!-- another publication -->

  <div class="row">
    <div class="col-md-6">
      <div class="pubimg">
        <img src="/assets/geology/shanxi.png" width="90%">
      </div>
    </div>
    <div class="col-md-6">
      <div class="pub">
        <div class="pubt">Taiyuan and Shanxi Formations Coal Rank Distribution and Metamorphism Analysis in Shanxi Province</div>
        <div class="pubd">Coals in The Shanxi Province China have complete coal ranks and excellent quality and the province is the main coking coal and anthracite base in the country. Based on coal resource prediction data and mining areas exploration data in many years, this paper summarizes the coal rank distribution and metamorphism features in Taiyuan and Shanxi Formations.
        </div>
        <div class="puba">Xichao Xie, Qinghui Zhang, Yuegang Tang, Zhengxi Zhang, Haisheng Wang, <b>Xianxu Hou</b>, Xiaodong Chen, Yufei Su, Cong Wang, Long Jia</div>
        <div class="pubv">Coal Geology of China, 2011</div>

        <div class="publ">
          <ul>
            <li><a href="/assets/geology/shanxi.pdf">[pdf]&nbsp;</a>(Chinese with English abstract)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<hr class="bold"/>

<center><h2>Other projects</h2></center>

<div>
  <div class="row">
    <div class="col-md-4">
      <div class="pets"><center>Robot Fire Fighting</center></div>
      <div>
        <a href="https://www.youtube.com/watch?v=Q1NCNPs0o1Y" target="_blank"><img src="/assets/pet-project/robotfire.gif" style="width:100%;border-radius:3px;"></a><br>
      </div>
    </div>
    <div class="col-md-4">
      <div class="pets"><center>Utopian Short Film</center></div>
      <a href="https://mediaspace.nottingham.ac.uk/media/Group4-Utopian/1_10b9r3a1" target="_blank"><img src="/assets/pet-project/Utopian.png" style="width:100%;border-radius:3px;"></a><br>
    </div>
   <div class="col-md-4">
      <div class="pets"><center>Dystopian Short Film</center></div>
      <a href="https://mediaspace.nottingham.ac.uk/media/Group4-Dystopian/1_247yqh4y" target="_blank"><img src="/assets/pet-project/dystopian.png" style="width:100%;border-radius:3px;"></a><br>
    </div>
  </div>

  <div class="row">
    <div class="col-md-4">
      <div class="pets"><center>Tetris</center></div>
      <div>
        <a href="http://houxianxu.herokuapp.com/project3_tetris/tetris.html" target="_blank"><img src="/assets/pet-project/tetris.png" style="width:100%;border-radius:3px;"></a><br>
      </div>
    </div>

  </div>
</div>


