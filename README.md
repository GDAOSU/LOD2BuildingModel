# Automated LoD-2 Model Reconstruction from Satellite-derived Digital Surface Model and Orthophoto
Shengxi, Gui, Rongjun, Qin  
ISPRS Journal of Photogrammetry and Remote Sensing
## Resources
[[PDF]](coming soon)  
[[CODE]](our code will coming soon)  
## Abstract
Digital surface models (DSM) generated from multi-stereo satellite images are getting higher in quality owning to the improved data resolution and photogrammetric reconstruction algorithms. Satellite images effectively act as a unique data source for 3D building modeling, because it provides a much wider data coverage with lower cost than the traditionally used LiDAR and airborne photogrammetry data. Although 3D building modeling from point clouds has been intensively investigated, most of the methods are still ad-hoc to specific types of buildings and require high-quality and high-resolution data sources as input. Therefore, when applied to satellite-based point cloud or DSMs, these developed approaches are not readily applicable and more adaptive and robust methods are needed. As a result, most of the existing work on building modeling from satellite DSM achieves LoD-1 generation. In this paper, we propose a model-driven method that reconstructs LoD-2 building models following a "decomposition-optimization-fitting” paradigm. The proposed method starts building detection results through a deep learning-based detector and vectorizes individual segments into polygons using a “three-step” polygon extraction method, followed by a novel grid-based decomposition method that decompose the complex and irregularly shaped building polygons to tightly combined elementary building rectangles ready to fit elementary building models. We have optionally introduced OpenStreetMap (OSM) and Graph-Cut (GC) labeling to further refine the orientation of 2D building rectangle. The 3D modeling step takes building-specific parameters such as hip lines, as well as non-rigid and regularized transformations to optimize the flexibility for using a minimal set of elementary models. Finally, roof type of building models are refined and adjacent building models in one building segment are merged into the complex polygonal model. Our proposed method have addressed a few technical caveats over existing methods, resulting in practically high-quality results, based on our evaluation and comparative study on a diverse set of experimental dataset of cities with different urban patterns. (codes /binaries may be made available under this GitHub page: https://github.com/GDAOSU/LOD2BuildingModel)
![workflow](https://user-images.githubusercontent.com/28346807/131371730-d22b7783-56aa-471c-b5aa-89858c9bd576.png)
![decomposition](https://user-images.githubusercontent.com/28346807/131371807-556fedbe-f7f6-4220-9814-0387d6a4678a.png)
![result1e](https://user-images.githubusercontent.com/28346807/131372200-ce1b5e2d-8382-4737-9491-1bda18c811c8.png)
![result2](https://user-images.githubusercontent.com/28346807/131372209-6319ef48-f592-4548-ad06-f5c28abeba5d.png)

## Conclusion
In this paper, we propose a LoD-2 model reconstruction approach performed on DSM and orthophoto derived from very high resolution mulit-view satellite stereo images (0.5 meter GSD). The proposed method follows a typical model-driven paradigm that follows a series of steps including: instance level building segment detection, initial 2D building polygon extraction, polygon decomposition and refinement, basic model fitting and merging, in which we address a few technical caveats over existing approaches: 1) we have deeply integrated the use of color and DSM information throughout the process to decide the polygonal extraction and decomposition to be context-aware (i.e., decision following orthophoto and DSM edges); 2) a grid-based decomposition approach to allow only horizontal and vertical scanning lines for computing gradient for regularized decompositions (parallelism and orthogonality). Six regions from two cities presenting various urban patterns are used for experiments and both IOU2 and IOU3 (for 2D and 3D evaluation) are evaluated, our approaches have achieved an IOU2 ranging from 47.12% to 80.85%, and an IOU3 ranging from 41.46% to 79.62%. Our comparative studies against a few state-of-the-art results suggested that our method achieves the best performance metrics in IOU measures and yields favorably visual results. Our parameter analysis indicates the robustness of threshold tuning for the proposed approach.

Given that our method assumes only a few model types rooted in rectangle shapes, the limitation is that the proposed approach may not perform for other types of buildings such as those with dome roofs and may to over-decompose complex-shaped buildings. It should be noted the proposed method involves a series of basic algorithms that may involve resolution-dependent parameters, and default values are set based on 0.5 meter resolution data and can be appropriately scaled when necessarily processing data with higher resolution, while the authors suggest when processing with higher resolution data that are potentially sourced from airborne platforms, bottom-up approaches or processing components can be potentially considered to yield favorable results. The proposed approach developed in this paper, is specifically designed for satellite-based data that rich the existing upper limit of resolution (0.3-0.5 GSD) to accommodate the data uncertainty and resolution at scale. In the region with numerous compact blocks, the proposed approach capability is limited to reconstruct the roof structure of those blocks.

In our future work, a direct prediction of model type and parameters will be attempted, and other building segmentation methods will be introduced for building mask improvement, and types of models will be increased rooted not only on rectangle shapes but also circular and complexly parameterized shapes, followed by continued investigation on approaches to favorably offer reasonable decomposition of overcomplex building and post-merging. In addition, as future works it is worth to establish benchmark datasets with varying sources, where LiDAR data are available to construct LoD-2 ground truth data, which can evaluate image-based building model reconstruction approaches.

## Reference
    (coming soon)
