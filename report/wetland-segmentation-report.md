# Wetland Segmentation Using Sentinel-2 Imagery and U-Net with Explainability Analysis

## Abstract
Wetlands are essential ecosystems, and accurate monitoring is critical for their management. This study evaluates a U-Net-based model for wetland segmentation using Sentinel-2 imagery in Ontario and compares it with a Random Forest baseline. Model performance was assessed using IoU, along with explainability analyses focusing on wetland types, spectral bands, and spatial features via Grad-CAM. U-Net achieved slightly higher accuracy and produced smoother predictions, while Random Forest showed comparable performance with faster training but noisier outputs. Explainability results revealed variation in detection across wetland types, the importance of spectral bands, and the role of network layers in capturing spatial patterns. While the results demonstrate strong potential, further evaluation under diverse temporal, seasonal, and spatial conditions is needed. Extending the approach to multiclass segmentation and broader datasets would enhance understanding of model performance.

## 1. Introduction
Wetlands are important natural resources that play crucial roles in maintaining environmental stability. They contribute to water quality improvement, carbon storage, methane emission reduction, flood mitigation, and storm regulation (Mitsch, et al., 2013; Smolders, S., et al., 2015; Ferreira, et al., 2023). Therefore, continuous preservation and effective management of wetlands are essential.

Monitoring wetlands is necessary to understand their current condition and temporal changes. In recent years, remote sensing technologies have advanced significantly and have been widely applied to wetland monitoring. Common approaches include mapping, classification, change detection, time series analysis, and prediction (Abdelmajeed, et al., 2023).

Among these approaches, image segmentation plays a key role by providing pixel-level identification of wetland areas. Unlike classification methods that assign a single label to an entire image or region, segmentation enables precise delineation of wetland boundaries and spatial patterns. This level of detail is particularly important for understanding heterogeneous environments such as wetlands, where different land cover types are often intermixed. However, this complexity also makes wetland segmentation a challenging task, as subtle spectral differences and mixed land cover can make accurate boundary detection difficult.

Traditionally, wetland segmentation has been performed using proprietary software such as eCognition (Grenier, M., et al., 2005; Moffett, et al., 2013). Subsequently, hybrid approaches combining segmentation and machine learning classification have been introduced to improve performance (Dronova, I., 2015). With the rapid advancement of deep learning, these methods have increasingly been applied to wetland segmentation tasks.

Starting from early convolutional neural network (CNN)-based models, including U-Net (Mohammadimanesh, et al., 2019; Cui, et al., 2019), a wide range of deep learning architectures have been developed for wetland segmentation. More recent models incorporate increasingly complex designs to further improve segmentation accuracy (Lin, et al., 2023).

Despite their strong performance, deep learning models are often difficult to interpret. Some studies have attempted to improve interpretability by incorporating feature selection or structured inputs (Gui, et al., 2025). However, interpretability in deep learning remains challenging and often depends on the data and methods used, and it generally does not reach the level of transparency provided by physically based models.

The objective of this study is to evaluate the effectiveness of deep learning methods for wetland segmentation and to investigate the interpretability (explainability) of the model.


## 2. Methodology
### 2.1. Task definition
Wetlands consist of multiple categories such as bog and swamp, and corresponding multi-class map data are available. However, this project focuses on binary segmentation for simplicity. Water surface segmentation is a well-studied problem and typically achieves high performance. In this study, not only water surfaces but all wetland areas are targeted for segmentation to examine how this broader definition increases task difficulty.

### 2.2. Wetland segmentation model
This project employs U-Net as the wetland segmentation model. U-Net, introduced in 2015, is a convolutional neural network (CNN)-based architecture widely used in computer vision and image segmentation tasks (Ronneberger, et al., 2015). Despite the development of more recent segmentation models, U-Net remains popular due to its simple structure and ease of implementation.

U-Net has a U-shaped architecture consisting of an encoder and a decoder. The encoder progressively reduces the spatial dimensions of the input while increasing the number of feature channels, allowing the model to capture high-level features. The decoder then restores the spatial resolution through upsampling and convolution operations, enabling precise localization. Convolutional layers are defined by kernel size and the number of output channels, while pooling layers are defined by kernel size and pooling type (e.g., max or average pooling).

In this study, a kernel size of 3 × 3 was used for all convolutional layers, as it is commonly adopted in U-Net architectures. The number of channels was designed to double in the encoder and halve in the decoder, following a standard configuration. Max pooling with a kernel size of 2 × 2 was selected to retain strong activations associated with edges and local contrasts, which are important for capturing spatial patterns in wetland imagery.

The input image size was set to 256 × 256 pixels, which is called a “patch”. This size was chosen because wetland features can be effectively represented at this resolution while maintaining a manageable computational cost. The input consists of six channels corresponding to selected spectral bands of the satellite imagery that are effective for wetland segmentation. Skip connections between the encoder and decoder are implemented by direct concatenation of feature maps at the same spatial resolution, without cropping, as the spatial dimensions are consistent across corresponding levels. The activation function used throughout the network is ReLU, which is commonly adopted in U-Net architectures.

This model can be characterized as a mathematical/computer, empirical, and stochastic model. It is data-driven, as it learns patterns directly from input data rather than relying on predefined physical equations. The model operates in a distributed manner, processing spatial information across all pixels through convolutional operations. It is primarily predictive, as it learns a mapping from input images to segmentation outputs. The model is static, as it operates on individual images without incorporating temporal dependencies. Figure 1 illustrates the overall U-Net architecture used in this study.

<p align="center">
  <img src="figures/fig1.png" width="600">
</p>
<p align="center">
  <em>Figure 1. The U-Net architecture used in this project.</em>
</p>

### 2.3. Data augmentation
Data augmentation is commonly used in image segmentation tasks to increase data diversity (Jin, et al., 2020). It consists of applying transformations to input data, such as rotation, flipping, scaling, color jittering, noise injection, and random erasing. This technique can improve model performance and robustness; however, it may also negatively affect learning if inappropriate transformations are applied.

Since this project focuses on achieving high performance with relatively clean imagery rather than robustness to noise, augmentations such as color jittering and noise injection were not applied. Although the training dataset already contains sufficient variability, additional experiments were conducted using simple geometric augmentations, including horizontal flipping, vertical flipping, and 90-degree rotation, each applied with a probability of 50%, to evaluate their impact on model performance.

### 2.4. Computer resources
The entire workflow, including model training, can be performed on a standard laptop; however, GPU acceleration significantly improves efficiency. Most experiments in this project were conducted using the Google Colab environment with an NVIDIA T4 GPU, which is available for free-use users under certain usage limits.

### 2.5. Performance evaluation
To quantify segmentation performance, Intersection over Union (IoU) was used as the primary evaluation metric, as it is widely adopted for image segmentation tasks (Jaccard, 1912). Since IoU alone may not fully capture segmentation quality, qualitative analysis was also performed by visually inspecting segmentation results on test images. In addition, non-functional performance metrics, including training time and final model size, were evaluated.

To enable comparison with a traditional approach, a Random Forest model was implemented as a baseline for the same task. Random Forest is a widely used machine learning method applicable to various problems, including image segmentation.

### 2.6. Model training
Dice loss (Milletari, et al., 2016) was used as the loss function for training the U-Net model, as it is closely related to IoU and well-suited for segmentation tasks. The stopping point for training was determined by monitoring the training Dice loss and validation IoU. A maximum of 40 epochs was set, as U-Net models typically reach optimal performance within this range.

Batch size is an important factor in deep learning training, as it determines the number of samples used for each parameter update. Larger batch sizes provide more stable gradient estimates but are limited by computational resources. In this study, a batch size of 16 was used, considering the memory capacity of the Google Colab T4 GPU.

### 2.7. Model explanability analysis

To understand how predictions are made by the trained model, explainability analysis was conducted. While deep learning models are not physically based and are often considered less interpretable, certain approaches can still provide insights into their behavior. For example, regression models offer interpretability through coefficients that indicate the contribution of each feature, and decision trees are inherently interpretable. In contrast, deep learning models typically sacrifice interpretability in exchange for their ability to automatically learn complex patterns from data. Recent explainable AI (XAI) methods aim to interpret deep learning models by identifying contributing factors in the input data. In this study, the input to U-Net consists of three key factors: (1) wetland types, (2) spectral bands, and (3) spatial information. Therefore, the model was analyzed from these three perspectives.

First, wetland type-wise analysis was conducted. It is assumed that detection difficulty varies across wetland types. Since this project adopts binary segmentation, IoU cannot be directly computed for each class. Instead, a coverage-based metric, similar to recall, was used. For each wetland type, the proportion of correctly predicted wetland pixels within the ground truth area of that type was calculated. A higher score indicates better detection performance for that wetland type:

$$
\text{Coverage}_c = \frac{TP_c}{GT_c} \quad (1)
$$

where:
- $TP_c$ denotes the number of pixels predicted as wetland that belong to class $c$  
- $GT_c$ denotes the total number of pixels of class $c$

Second, band-wise analysis was performed. Since the input images contain multiple spectral bands, each band may contribute differently to the prediction. To evaluate this, band occlusion analysis was applied by removing one band at a time from the input. The impact was measured using IoU drop, defined as the decrease in IoU compared to the full-band input. This analysis was conducted for each band to assess its relative importance.

Finally, geometry-wise (spatial) analysis was conducted using Grad-CAM (Selvaraju, et al., 2016). Grad-CAM is an explainability method that leverages gradients of feature maps to identify regions in the input that strongly influence model predictions. When applied to U-Net, it highlights areas where the prediction is most sensitive to changes in feature values across spatial locations. The target layer can be selected from different stages of the U-Net architecture (e.g., down1, bottleneck, conv4), as illustrated in Figure 1. In image segmentation tasks, regions near object boundaries often exhibit stronger responses. This method was used to qualitatively assess how the model captures spatial patterns in wetland areas.

### 2.8. User interface for prediction experiments
A graphical user interface (GUI) was developed to facilitate prediction and visualization. The intended use case assumes that a trained U-Net model is provided, allowing users to select an input image, corresponding ground truth, a Grad-CAM target layer, and an output file path. The tool generates a single figure that includes the original image, ground truth, prediction output, and Grad-CAM visualization, and optionally saves the figure to disk. GUI was developed using tkinter library. An example of the GUI is shown in Figure 2.

<p align="center">
  <img src="figures/fig2.png" width="300">
</p>
<p align="center">
  <em>Figure 2. GUI input window.</em>
</p>

## 3. Methodology
### 3.1. Data requirements
Since the primary objective of this project is to evaluate the fundamental segmentation capability of the model, I wanted to minimize potential limitations caused by data quality as much as possible. In addition, using recent data is desirable to ensure applicability to current environmental conditions.

Sentinel-2 imagery was selected as the primary data source. It provides high spatial resolution (10 m), a wide range of spectral bands, and up-to-date global coverage. The bands used in this study are B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR-1), and B12 (SWIR-2). These bands capture information related to water presence, vegetation condition, and soil moisture, which are essential for wetland detection. All satellite imagery was acquired using Google Earth Engine (https://earthengine.google.com/).

Regarding acquisition dates, it is important to align the imagery with the ground truth data. In addition, seasonal variation significantly affects landscape appearance. Summer was selected as the target season because vegetation is fully developed and there is no interference from snow cover. Based on these considerations, images were filtered within the period from June 15, 2021 to September 15, 2021.

Cloud coverage was also considered during data selection. While lower cloud coverage improves image quality, it reduces spatial availability. After preliminary experiments, a threshold of less than 20% cloud coverage was chosen as a balance between image quality and spatial coverage.

Ground truth data were obtained from the Canadian Wetland Inventory Map Version 3A (CWIM3A) provided by GEO.ca (https://geo.ca/). CWIM3A includes both wetland extent and five wetland classes: bog, fen, swamp, marsh, and water. To align the ground truth with satellite imagery, the CWIM3A raster data were reprojected to the same coordinate reference system as the satellite images and then cropped to match the extent of each image tile, ensuring pixel-level alignment.

### 3.2 Study areas
Wetlands are widely distributed across the globe and are present on all continents (Sharma, et al., 2021). Their global extent is estimated to exceed 1,800 million hectares (Dudley, 2025). Canada contains abundant and diverse wetland ecosystems.

From a data requirement perspective, approximately 500 image patches for training and 100 for testing were considered sufficient. Given that each patch is 256 × 256 pixels with a spatial resolution of 10 m, this corresponds to an area of approximately 4,000 km², which is about 5% of the total area of Ontario. Considering its diverse landscapes and sufficient wetland coverage, Ontario was selected as the study region.

Within Ontario, Southern Ontario contains a variety of land cover types, including forests, agricultural areas, and urban regions, often interspersed with wetlands. Therefore, study areas were primarily selected from Southern Ontario, with additional samples from Eastern and Northern Ontario to ensure diversity across training and testing datasets.

Since U-Net requires a fixed input size of 256 × 256 pixels, this constraint was considered during data collection. Satellite imagery was acquired using the Google Earth Engine API; however, specifying geographic extents does not guarantee fixed image dimensions. To address this, study areas were selected such that they could be divided into an integer number of 256 × 256 patches. Buffer regions were applied uniformly in both spatial dimensions due to API limitations. As a result, the final study areas consist of square regions with slight margins.

Large open water bodies, such as Lake Ontario and Lake Huron, are not classified as wetlands in the ground truth data and were therefore excluded from the study areas. Additionally, spatial distortions and inconsistencies in data acquisition required iterative adjustments to the selected regions. After several rounds of data collection and refinement, a total of 580 patches for training and 186 patches for testing were obtained. Figure 3 illustrates the spatial distribution of the sampled images across Ontario, and detailed data collection statistics are provided in Appendix A: Study area selection.

<p align="center">
  <img src="figures/fig3.png" width="600">
</p>
<p align="center">
  <em>Figure 3. Wetland types in Ontario and study areas.</em>
</p>

## 4. Results
### 4.1. Model comparisons
During U-Net training without data augmentation, the highest validation IoU of 0.750 was achieved at the 39th epoch. When data augmentation was applied, the best model was obtained at the 37th epoch with a validation IoU of 0.742. For the baseline model, Random Forest, the optimal hyperparameters were determined through grid search as 200 trees, a maximum depth of 20, and a minimum of 2 samples per leaf, resulting in a validation IoU of 0.746.

The transitions of training loss and validation IoU for both models (with and without augmentation) are shown in Figure 4. Rapid improvement was observed within the first 10 epochs, followed by a slower increase. No clear decline in validation IoU was observed before 40 epochs. While training loss continued to decrease gradually, no strong convergence was observed. 

The test IoU scores are included in Figure 5 as well. U-Net with data augmentation achieved the highest performance with an IoU of 0.661. However, the differences among the models were relatively small. Sample segmentation results are shown in Figure 5. The Random Forest output exhibits many isolated noisy pixels, whereas U-Net produces smoother and more spatially consistent boundaries.

Non-functional performance comparisons are presented in Figure 6. Training time was measured for model training only, excluding hyperparameter tuning. The Random Forest model was trained in approximately 3 minutes, whereas U-Net required around 20 minutes. In terms of model size, the Random Forest model occupies approximately 500 MB when saved, while the U-Net model is significantly smaller at approximately 120 MB. 

<p align="center">
  <img src="figures/fig4.png" width="300">
</p>
<p align="center">
  <em>Figure 4. Transition of training loss and validation IoU (U-Net + data augmentation).</em>
</p>

<p align="center">
  <img src="figures/fig5.png" width="600">
</p>
<p align="center">
  <em>Figure 5. Segmentation results for a study area in Bruce Peninsula National Park using three models.</em>
</p>

<p align="center">
  <img src="figures/fig6.png" width="200">
</p>
<p align="center">
  <em>Figure 6. Model comparison: training time and model size.</em>
</p>

<p align="center">
  <img src="figures/fig7.png" width="200">
</p>
<p align="center">
  <em>Figure 7. Coverage rate by wetland type.</em>
</p>

<p align="center">
  <img src="figures/fig8.png" width="200">
</p>
<p align="center">
  <em>Figure 8. Band importance from band occlusion analysis.</em>
</p>

### 4.2 Explainability analysis
Among the two U-Net models, the version with data augmentation was selected for explainability analysis, as it achieved the highest test IoU.

The first analysis focuses on wetland type-wise performance. As shown in Figure 7, water areas are almost perfectly detected by the model. Coverage rates for other wetland types vary considerably. Bog shows a high coverage rate of approximately 92%, whereas marsh is poorly detected, with less than half of its area correctly predicted as wetland.

Band-wise analysis results are shown in Figure 8. All bands exhibit positive IoU drops when occluded. Among them, B2 (Blue), B3 (Green), and B8 (NIR) show the largest IoU drops (greater than 0.27).

Grad-CAM results for a sample area are shown in Figure 9. This visualization illustrates how activation patterns evolve across layers of the U-Net during the forward pass. The first encoder layer primarily captures the boundaries of wetland areas. As the network progresses to deeper layers, the activations shift toward regions within these boundaries. In the bottleneck layer, responses become more concentrated, reflecting highly abstracted feature representations. During the decoding process, spatial details are gradually restored, leading to the final segmentation output.

<p align="center">
  <img src="figures/fig9.png" width="600">
</p>
<p align="center">
  <em>Figure 9. Grad-CAM visualization for a study area in Bruce Peninsula National Park by layer.</em>
</p>

## 5. Discussion
### 5.1. Model validity and advantages
Due to computational constraints, training could not be extended indefinitely, and the maximum number of epochs was set to 40. As both the increase in validation IoU and the decrease in training loss had already begun to plateau, further training was unlikely to yield significant improvements and could potentially lead to overfitting. Therefore, the selected U-Net models trained within 40 epochs are considered appropriate.

In terms of model comparison, the three models showed similar performance based on test IoU scores. However, qualitative inspection revealed differences in their behavior. The Random Forest model produced outputs with many isolated noisy pixels, suggesting a tendency toward overfitting, whereas U-Net generated smoother segmentation results, potentially oversimplifying boundaries. From a purely quantitative perspective, it is difficult to conclude that one model clearly outperforms the other. Therefore, model selection may depend more on usability and practical considerations. Random Forest is conceptually easier to interpret, but it requires extensive hyperparameter tuning and can result in large model sizes depending on the configuration.

The necessity of data augmentation was not strongly supported by the results. In this study, a sufficient amount and diversity of training data were available, reducing the need for augmentation. However, data augmentation may become important when dealing with limited datasets or lower-quality imagery. In this experiment, augmentation did not negatively impact model performance.

The wetland type-wise analysis revealed notable differences in coverage rates across classes. These variations may be caused by class imbalance in the dataset or insufficient feature representation for certain wetland types. Further investigation is required to identify the underlying causes.

Regarding band importance, all bands contributed to the model prediction, as indicated by positive IoU drops in the occlusion analysis. The higher importance of the blue, green, and NIR bands is consistent with their roles in capturing water characteristics, vegetation properties, and spectral contrast. However, this observation may also be influenced by data distribution, and further analysis is needed.

The geometry-wise analysis using Grad-CAM illustrates how the model processes spatial patterns across layers. The first encoder layer primarily focuses on the boundaries of water areas, while subsequent layers also highlight other wetland types. This suggests that different layers may capture features relevant to different wetland classes. The representations in the bottleneck layer are more abstract and therefore difficult to interpret directly. In other tasks, Grad-CAM visualizations may provide more intuitive interpretations. For example, in image classification, Grad-CAM typically highlights the most discriminative regions contributing to the predicted class. In this study, the visualization indicates that the model relies on spatial patterns, particularly boundaries and mixed regions, for wetland segmentation.

### 5.2. Limitations
The satellite imagery used in this study was limited to the year 2021 to align with the ground truth data. While this allows for accurate evaluation under consistent conditions, it does not guarantee that the model will generalize well to more recent data. Testing with newer imagery (e.g., 2025) would be desirable. However, exact spatial alignment with the original dataset might be difficult to achieve.

Seasonal variation is another important factor that may affect model performance. This study focused on summer imagery due to clearer conditions, including fully developed vegetation and the absence of snow. However, model performance may differ when applied to images from other seasons. For broader applicability, evaluation across multiple seasons is necessary.

The model also imposes several practical constraints. For example, U-Net requires a fixed input size (256 × 256 pixels), which necessitates additional preprocessing and postprocessing when applied to images of arbitrary size. Furthermore, although this study used 10 m resolution imagery, the model’s performance on coarser-resolution data remains uncertain and should be evaluated for more general use cases.

Finally, the task was defined as binary segmentation for simplicity. While this approach simplifies the problem, it limits the ability to analyze differences among wetland types. To improve both functionality and segmentation quality, future work should consider multiclass segmentation.


## 6. Conclusion
This study evaluated the effectiveness of U-Net for wetland segmentation using Sentinel-2 imagery in Ontario, with comparisons to a Random Forest baseline. The results showed that U-Net achieved slightly higher performance and produced smoother and more spatially consistent predictions, while Random Forest provided comparable accuracy with faster training but noisier outputs and a larger model size. Explainability analyses revealed that model performance varies across wetland types and highlighted the importance of spectral bands. Grad-CAM further demonstrated how the model processes input data across layers. Although the model performed well under controlled conditions, its generalizability across time, season, and spatial resolution remains a limitation. Future work should explore multiclass segmentation and broader validation to improve both performance and applicability.

## References
[1] Mitsch, W. J., Bernal, B., Nahlik, A. M., Mander, Ü., Zhang, L., Anderson, C. J., ... & Brix, H. (2013). Wetlands, carbon, and climate change. Landscape ecology, 28(4), 583-597.\
[2] Smolders, S., Plancke, Y., Ides, S., Meire, P., & Temmerman, S. (2015). Role of intertidal wetlands for tidal and storm tide attenuation along a confined estuary: a model study. Natural Hazards and Earth System Science, 15(7), 1659-1675.\
[3] Ferreira, C. S., Kašanin-Grubin, M., Solomun, M. K., Sushkova, S., Minkina, T., Zhao, W., & Kalantari, Z. (2023). Wetlands as nature-based solutions for water management in different environments. Current Opinion in Environmental Science & Health, 33, 100476. \
[4] Abdelmajeed, A. Y. A., Albert-Saiz, M., Rastogi, A., & Juszczak, R. (2023). Cloud-based remote sensing for wetland monitoring—A review. Remote Sensing, 15(6), 1660. \
[5] Grenier, M., Demers, A. M., Labrecque, S., Fournier, R. A., Drolet, B., & Benoit, M. (2005). A classification method to map wetlands in Quebec for the Canadian Wetland Inventory using a top-down approach with object-oriented segmentation. \
[6] Moffett, K. B., & Gorelick, S. M. (2013). Distinguishing wetland vegetation and channel features with object-based image segmentation. International Journal of Remote Sensing, 34(4), 1332-1354. \
[7] Dronova, I. (2015). Object-based image analysis in wetland research: a review. Remote Sens 7: 6380–6413. \
[8] Mohammadimanesh, F., Salehi, B., Mahdianpari, M., Gill, E., & Molinier, M. (2019). A new fully convolutional neural network for semantic segmentation of polarimetric SAR imagery in complex land cover ecosystem. ISPRS journal of photogrammetry and remote sensing, 151, 223-236. \
[9] Cui, B., Zhang, Y., Li, X., Wu, J., & Lu, Y. (2019, November). WetlandNet: semantic segmentation for remote sensing images of coastal wetlands via improved UNet with deconvolution. In International conference on genetic and evolutionary computing (pp. 281-292). Singapore: Springer Singapore. \
[10] Lin, X., Cheng, Y., Chen, G., Chen, W., Chen, R., Gao, D., ... & Wu, Y. (2023). Semantic segmentation of China’s coastal wetlands based on Sentinel-2 and Segformer. Remote Sensing, 15(15), 3714. \
[11] Gui, B., Sam, L., Bhardwaj, A., Gómez, D. S., Peñaloza, F. G., Buchroithner, M. F., & Green, D. R. (2025). SAGRNet: A novel object-based graph convolutional neural network for diverse vegetation cover classification in remotely-sensed imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 227, 99-124. \
[12] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Cham: Springer international publishing. \
[13] Jin, Y. W., Jia, S., Ashraf, A. B., & Hu, P. (2020). Integrative data augmentation with U-Net segmentation masks improves detection of lymph node metastases in breast cancer patients. Cancers, 12(10), 2934. \
[14] Jaccard, P. (1912). The distribution of the flora in the alpine zone. 1. New phytologist, 11(2), 37-50. \
[15] Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV) (pp. 565-571). Ieee. \
[16] Selvaraju, R. R., Das, A., Vedantam, R., Cogswell, M., Parikh, D., & Batra, D. (2016). Grad-CAM: Why did you say that?. arXiv preprint arXiv:1611.07450. \
[17] Sharma, S., & Singh, P. (Eds.). (2021). Wetlands conservation: Current challenges and future strategies. John Wiley & Sons. \
[18] Dudley, N. (Ed.). (2025). Global Wetland Outlook 2025: Valuing, Conserving, Restoring and Financing Wetlands. Secretariat of the Convention on Wetlands.

## Appendices
### Appendix A: Study area selection
This section shows the whole information of study area selection.

<p align="center">
  <em>Table 1. Sampled study areas in Ontario.</em>
</p>

| ID  | Purpose                        | Usage | Location                     | Latitude | Longitude | Train Patches | Test Patches |
|-----|--------------------------------|--------|------------------------------|----------|-----------|---------------|--------------|
| 001 | Wetland/Urban                 | Train  | Sir Adam Beck (Lake)        | 43.14    | -79.06    | 1             | -            |
| 002 | Wetland/Urban                 | Train  | Thorold                      | 43.10    | -79.22    | 1             | -            |
| 003 | Forest/Wetland                | Train  | Port Colborne                | 42.92    | -79.26    | 1             | -            |
| 004 | Forest                        | Test   | Port Colborne                | 42.91    | -79.33    | -             | 1            |
| 005 | Wetland/Urban/Forest          | Train  | Dunnville                    | 42.91    | -79.63    | 0             | -            |
| 006 | Wetland/Urban/Agriculture     | Train  | Cayuga                       | 42.95    | -79.86    | 4             | -            |
| 007 | Wetland/Agriculture/Forest    | Train  | Ohsweken                     | 43.10    | -80.07    | 4             | -            |
| 008 | Wetland/Agriculture/Forest    | Test   | Ohsweken                     | 43.08    | -80.17    | -             | 1            |
| 009 | Forest                        | Train  | Ohsweken                     | 43.04    | -80.05    | 1             | -            |
| 010 | Forest                        | Test   | Ohsweken                     | 43.05    | -80.17    | -             | 1            |
| 011 | Wetland/Urban                 | Train  | Hamilton - Land              | 43.37    | -79.92    | 0             | -            |
| 012 | Wetland/Urban                 | Train  | Hamilton - Shore             | 43.22    | -79.89    | 1             | -            |
| 013 | Wetland/Urban                 | Test   | Burlington                   | 43.21    | -79.80    | -             | 1            |
| 014 | Urban/Wetland/Agriculture     | Train  | Cambridge                    | 43.44    | -80.40    | 4             | -            |
| 015 | Urban/Wetland/Agriculture     | Test   | Waterloo                     | 43.48    | -80.57    | -             | 1            |
| 016 | Agriculture/Wetland           | Train  | Port Royal                   | 42.64    | -80.54    | 4             | -            |
| 017 | Agriculture/Wetland           | Test   | Turkey Point                 | 42.69    | -80.37    | -             | 1            |
| 018 | Peninsula                     | Train  | The Cottages                 | 42.58    | -80.31    | 0             | -            |
| 019 | Peninsula                     | Test   | The Cottages                 | 42.57    | -80.27    | -             | 0            |
| 020 | Peninsula                     | Train  | Long Point National Area     | 42.55    | -80.12    | 0             | -            |
| 021 | Peninsula                     | Test   | Long Point National Area     | 42.55    | -80.16    | -             | 0            |
| 022 | Agriculture                   | Train  | Petrolia                     | 42.80    | -82.23    | 4             | -            |
| 023 | Agriculture                   | Test   | Petrolia                     | 42.76    | -81.96    | -             | 1            |
| 024 | Peninsula                     | Train  | Walpole Island               | 42.54    | -82.53    | 1             | -            |
| 025 | Peninsula                     | Test   | Walpole Island               | 42.53    | -82.48    | -             | 1            |
| 026 | Agriculture/Wetland           | Train  | Flesherton                   | 44.25    | -80.49    | 9             | -            |
| 027 | Agriculture/Wetland           | Test   | Flesherton                   | 44.35    | -80.28    | -             | 4            |
| 028 | Peninsula                     | Train  | Bruce Peninsula              | 45.13    | -81.42    | 4             | -            |
| 029 | Peninsula                     | Test   | Bruce Peninsula              | 45.21    | -81.53    | -             | 1            |
| 030 | Peninsula                     | Train  | Wolseley                     | 44.67    | -81.04    | 4             | -            |
| 031 | Peninsula                     | Test   | Mar                          | 44.80    | -81.22    | -             | 1            |
| 032 | Urban/Wetland                 | Train  | Toronto                      | 43.70    | -79.42    | 4             | -            |
| 033 | Urban/Wetland                 | Test   | Toronto                      | 43.70    | -79.31    | -             | 1            |
| 034 | Urban/Wetland                 | Train  | Mississauga                  | 43.61    | -79.72    | 4             | -            |
| 035 | Urban/Wetland                 | Test   | Mississauga                  | 43.71    | -79.72    | -             | 1            |
| 036 | Wetland                       | Train  | Lake Simcoe                  | 44.23    | -79.56    | 12            | -            |
| 037 | Wetland                       | Test   | Lake Simcoe                  | 44.34    | -79.27    | -             | 4            |
| 038 | Wetland                       | Train  | Curve Lake                   | 44.47    | -78.54    | 16            | -            |
| 039 | Wetland                       | Test   | Curve Lake                   | 44.60    | -78.85    | -             | 9            |
| 040 | Wetland                       | Train  | Campbellford                 | 44.24    | -77.76    | 16            | -            |
| 041 | Wetland                       | Train  | Thomasburg                   | 44.40    | -77.21    | 0             | -            |
| 042 | Wetland                       | Test   | Thomasburg                   | 44.36    | -77.15    | -             | 0            |
| 043 | Island                        | Train  | Allisonville                 | 43.97    | -77.20    | 1             | -            |
| 044 | Island                        | Test   | Allisonville                 | 44.02    | -77.43    | -             | 1            |
| 045 | Wetland                       | Train  | Smiths Falls                 | 44.90    | -75.90    | 16            | -            |
| 046 | Wetland                       | Test   | Smiths Falls                 | 44.86    | -76.13    | -             | 4            |
| 047 | Urban/Wetland                 | Train  | Ottawa                       | 45.41    | -75.60    | 16            | -            |
| 048 | Urban/Wetland                 | Test   | Ottawa                       | 45.45    | -75.42    | -             | 4            |
| 049 | Wetland                       | Test   | Westmeath                    | 45.83    | -76.86    | -             | 1            |
| 050 | Island (Wetland)              | Train  | Manitoulin Island            | 45.80    | -82.28    | 9             | -            |
| 051 | Island (Wetland)              | Test   | Manitoulin Island            | 45.79    | -81.95    | -             | 1            |
| 052 | Wetland                       | Train  | Peawanuk                     | 54.57    | -84.65    | 8             | -            |
| 053 | Wetland                       | Test   | Peawanuk                     | 54.41    | -84.66    | -             | 1            |