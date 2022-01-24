# Automated LoD-2 Model Reconstruction from Satellite-derived Digital Surface Model and Orthophoto
Shengxi, Gui, Rongjun, Qin  
ISPRS Journal of Photogrammetry and Remote Sensing
## Resources
[[PDF]]https://arxiv.org/ftp/arxiv/papers/2109/2109.03876.pdf  
[[CODE]]https://github.com/GDAOSU/LOD2BuildingModel  
## Abstract
Digital surface models (DSM) generated from multi-stereo satellite images are getting higher in quality owning to the improved data resolution and photogrammetric reconstruction algorithms. Satellite images effectively act as a unique data source for 3D building modeling, because it provides a much wider data coverage with lower cost than the traditionally used LiDAR and airborne photogrammetry data. Although 3D building modeling from point clouds has been intensively investigated, most of the methods are still ad-hoc to specific types of buildings and require high-quality and high-resolution data sources as input. Therefore, when applied to satellite-based point cloud or DSMs, these developed approaches are not readily applicable and more adaptive and robust methods are needed. As a result, most of the existing work on building modeling from satellite DSM achieves LoD-1 generation. In this paper, we propose a model-driven method that reconstructs LoD-2 building models following a "decomposition-optimization-fitting” paradigm. The proposed method starts building detection results through a deep learning-based detector and vectorizes individual segments into polygons using a “three-step” polygon extraction method, followed by a novel grid-based decomposition method that decompose the complex and irregularly shaped building polygons to tightly combined elementary building rectangles ready to fit elementary building models. We have optionally introduced OpenStreetMap (OSM) and Graph-Cut (GC) labeling to further refine the orientation of 2D building rectangle. The 3D modeling step takes building-specific parameters such as hip lines, as well as non-rigid and regularized transformations to optimize the flexibility for using a minimal set of elementary models. Finally, roof type of building models are refined and adjacent building models in one building segment are merged into the complex polygonal model. Our proposed method have addressed a few technical caveats over existing methods, resulting in practically high-quality results, based on our evaluation and comparative study on a diverse set of experimental dataset of cities with different urban patterns. (codes /binaries may be made available under this GitHub page: https://github.com/GDAOSU/LOD2BuildingModel)
![workflow](https://user-images.githubusercontent.com/28346807/131371730-d22b7783-56aa-471c-b5aa-89858c9bd576.png)
![decomposition](https://user-images.githubusercontent.com/28346807/131371807-556fedbe-f7f6-4220-9814-0387d6a4678a.png)
![result1e](https://user-images.githubusercontent.com/28346807/131372200-ce1b5e2d-8382-4737-9491-1bda18c811c8.png)
![result2](https://user-images.githubusercontent.com/28346807/131372209-6319ef48-f592-4548-ad06-f5c28abeba5d.png)

## TODO

- Building LoD-2 generation via satellite data.


## Usage
### SAT2LoD2: LoD-2 building model reconstruction software

Please refer to the user manual in ./softawre/software_user_manual.pdf
The sampel input and output files are at ./softawre/example/
The executable file is at ./softawre/SAT2LoD2/SAT2LoD2.exe


### open source code of LoD-2 model reconstruction

Python is the only language for software and open-source codes.
Please refer to the code in ./code/
./code/SAT2LoD2.py is the software GUI and main function of whole workflow;
./code/hrnet_seg.py corresponses to Building detection and segmentation;
./code/building_polygon.py corresponses to Initial 2D building polygon extraction;
./code/building_decomposition.py corresponses to Building rectangle decomposition;
./code/building_refinement.py corresponses to Building rectangle orientation refinement;
./code/building_modelfit.py corresponses to 3D model fitting;
./code/building_obj.py corresponses to Mesh generation.

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
```
