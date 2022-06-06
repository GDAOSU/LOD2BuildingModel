# Automated LoD-2 Model Reconstruction from Satellite-derived Digital Surface Model and Orthophoto

This is a LoD-2 building model reconstruction approach with code, also the software version called "SAT2LoD2".  
  
The journal paper [Automated LoD-2 Model Reconstruction from Satellite-derived Digital Surface Model and Orthophoto](https://www.sciencedirect.com/science/article/pii/S0924271621002318) is the technical description of the approach.  

  
The conference paper (accepted) [Sat2LoD2: A Software For Automated Lod-2 Modeling From Satellite-Derived Orthophoto And Digital Surface Model](https://arxiv.org/abs/2204.04139) describing the tools (SAT2LoD2) which achieve by Python, operations and with some slight updates.  

  
## Resources
[[(PDF) Automated LoD-2 Model Reconstruction from Satellite-derived Digital Surface Model and Orthophoto](https://arxiv.org/ftp/arxiv/papers/2109/2109.03876.pdf)]  
[[(PDF) Sat2lod2: A Software For Automated Lod-2 Modeling From Satellite-Derived Orthophoto And Digital Surface Model](https://arxiv.org/ftp/arxiv/papers/2204/2204.04139.pdf)]
## Abstract
Digital surface models (DSM) generated from multi-stereo satellite images are getting higher in quality owning to the improved data resolution and photogrammetric reconstruction algorithms. Satellite images effectively act as a unique data source for 3D building modeling, because it provides a much wider data coverage with lower cost than the traditionally used LiDAR and airborne photogrammetry data. Although 3D building modeling from point clouds has been intensively investigated, most of the methods are still ad-hoc to specific types of buildings and require high-quality and high-resolution data sources as input. Therefore, when applied to satellite-based point cloud or DSMs, these developed approaches are not readily applicable and more adaptive and robust methods are needed. As a result, most of the existing work on building modeling from satellite DSM achieves LoD-1 generation. In this paper, we propose a model-driven method that reconstructs LoD-2 building models following a "decomposition-optimization-fitting” paradigm. The proposed method starts building detection results through a deep learning-based detector and vectorizes individual segments into polygons using a “three-step” polygon extraction method, followed by a novel grid-based decomposition method that decompose the complex and irregularly shaped building polygons to tightly combined elementary building rectangles ready to fit elementary building models. We have optionally introduced OpenStreetMap (OSM) and Graph-Cut (GC) labeling to further refine the orientation of 2D building rectangle. The 3D modeling step takes building-specific parameters such as hip lines, as well as non-rigid and regularized transformations to optimize the flexibility for using a minimal set of elementary models. Finally, roof type of building models are refined and adjacent building models in one building segment are merged into the complex polygonal model. Our proposed method have addressed a few technical caveats over existing methods, resulting in practically high-quality results, based on our evaluation and comparative study on a diverse set of experimental dataset of cities with different urban patterns. 

![workflow](https://user-images.githubusercontent.com/32317924/151380594-1a349c74-93ce-493a-b7cc-75aeb0076bdb.jpg)
![decomposition](https://user-images.githubusercontent.com/32317924/151380692-9d94ae99-2df0-4515-bf4f-f3acc35c696f.jpg)
![result1e](https://user-images.githubusercontent.com/32317924/151380719-2166784d-1454-426b-87ff-df69db505a0f.jpg)
![resultfigure1](https://user-images.githubusercontent.com/28346807/172088779-8a0f103c-f141-478c-bc5e-21ea9e656f3b.PNG)


## TODO

- Building LoD-2 generation via satellite data.  


## Usage
### SAT2LoD2: LoD-2 building model reconstruction software
  
Please refer to the user manual in ./softawre/software_user_manual.pdf  
The sampel input and output files are at ./softawre/example/  
The executable file is at ./softawre/SAT2LoD2/SAT2LoD2.exe  
  
Operation video is available in Youtube: [SAT2LoD2 operation video](https://youtu.be/Nn4OABsEOXk)  
  
There are two individual version of SAT2LoD2 for PC with Nvidia CUDA 10 driver and CUDA 11 driver:  
If your GPU is RTX 30 series, please download the version for CUDA 11:   
- Onedrive: [SAT2LoD2 CUDA11 Onedrive](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/qin_324_osu_edu/EUAersQxRFpHtZPKUfnXTXYBXOn575uHZ97cdEZ_Wx_LHw?e=qaPbw9)  
- Google drive: [SAT2LoD2 CUDA11 Google drive](https://drive.google.com/file/d/1rA7SRPbSYFJwOBc7IfXxBgmUroTOZIOF/view?usp=sharing)  
  
If your GPU is in other situation, please download the version for CUDA 10:   
- Onedrive: [SAT2LoD2 CUDA10 Onedrive](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/qin_324_osu_edu/EWAmq2Rmr-lHuU9C6fRzsjcBJ7WvB0DRbXArBmXRaGt79w?e=1FyGGf)  
- Google drive: [SAT2LoD2 CUDA10 Google drive](https://drive.google.com/file/d/1t_4epCmaXPuZgHz3QenU-zfd7-vqrvKV/view?usp=sharing)  
  
  
#### Notice: SAT2LoD2 software and open-source has minor improvments compare with the pulished ISPRS Journal paper include: 1) building segmentation based on HRNetV2; 2) non-rectangular shape buildings mesh generation. Moreover, processing time for software and code is optimized as well.  
  
    
### Open source code of LoD-2 model reconstruction
  
Python is the only language for software and open-source codes.  
Please refer to the code in ./code/;  
./code/SAT2LoD2.py is the software GUI and main function of whole workflow;  
./code/hrnet_seg.py corresponses to Building detection and segmentation;  
./code/building_polygon.py corresponses to Initial 2D building polygon extraction;  
./code/building_decomposition.py corresponses to Building rectangle decomposition;  
./code/building_refinement.py corresponses to Building rectangle orientation refinement;  
./code/building_modelfit.py corresponses to 3D model fitting;  
./code/building_obj.py corresponses to Mesh generation.  

The weight for building segmentation download here:   
- Onedrive: [HRNet_wight Onedrive](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/qin_324_osu_edu/EXWTkwmtb1lPqnkbo57Ttn8BfHcIFfzQPZ41naZBNO8rZA?e=Roocco)  
- Google drive: [HRNet_wight Google drive](https://drive.google.com/file/d/1ya8_t67mpYJl1E7_1GyqSgcBgzIgGQg1/view?usp=sharing)

## Reference
```
@article{gui2021automated,
  title={Automated LoD-2 model reconstruction from very-high-resolution satellite-derived digital surface model and orthophoto},
  author={Gui, Shengxi and Qin, Rongjun},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={181},
  pages={1--19},
  year={2021},
  publisher={Elsevier}
}
@article{gui2022sat2lod2,
  title={Sat2lod2: A Software For Automated Lod-2 Modeling From Satellite-Derived Orthophoto And Digital Surface Model},
  author={Gui, Shengxi and Qin, Rongjun and Tang, Yang},
  journal={arXiv preprint arXiv:2204.04139},
  year={2022}
}
```
