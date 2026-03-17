---
library_name: transformers
tags:
- keypoint-matching
license: apache-2.0
---

# EfficientLoFTR

The Efficient LoFTR model was proposed in "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" by Yifan Wang, Xingyi He, Sida Peng, Dongli Tan, and Xiaowei Zhou from Zhejiang University.

This model presents a novel method for efficiently producing semi-dense matches across images, addressing the limitations of previous detector-free matchers like LoFTR, which suffered from low efficiency despite remarkable matching capabilities in challenging scenarios. Efficient LoFTR revisits design choices to improve both efficiency and accuracy.

The abstract from the paper is the following:

"We present a novel method for efficiently producing semi-dense matches across images. Previous detector-free matcher LOFTR has shown remarkable matching capability in handling large-viewpoint change and texture-poor scenarios but suffers from low efficiency. We revisit its design choices and derive multiple improvements for both efficiency and accuracy. One key observation is that performing the transformer over the entire feature map is redundant due to shared local information, therefore we propose an aggregated attention mechanism with adaptive token selection for efficiency. Furthermore, we find spatial variance exists in LoFTR's fine correlation module, which is adverse to matching accuracy. A novel two-stage correlation layer is proposed to achieve accurate subpixel correspondences for accuracy improvement. Our efficiency optimized model is ~2.5 faster than LoFTR which can even surpass state-of-the-art efficient sparse matching pipeline SuperPoint + LightGlue. Moreover, extensive experiments show that our method can achieve higher accuracy compared with competitive semi-dense matchers, with considerable efficiency benefits. This opens up exciting prospects for large-scale or latency-sensitive applications such as image retrieval and 3D reconstruction. Project page: https://zju3dv.github.io/efficientloftr/" 


![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/WQXZvIqWcXhtmw-mmcPE3.png)

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/zju3dv/efficientloftr).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Efficient LoFTR is a neural network designed for semi-dense local feature matching across images, building upon and significantly improving the detector-free matcher LoFTR. The key innovations include:

An aggregated attention mechanism with adaptive token selection for efficient feature transformation, addressing the redundancy of performing transformers over the entire feature map due to shared local information. This mechanism significantly reduces the cost of local feature transformation by aggregating features for salient tokens and utilizing vanilla attention with relative positional encoding.

A novel two-stage correlation layer for accurate subpixel correspondence refinement. This module first locates pixel-level matches using mutual-nearest-neighbor (MNN) matching on fine feature patches and then refines them for subpixel accuracy by performing correlation and expectation locally within tiny patches, thereby addressing spatial variance observed in LoFTR's refinement phase.

The model is designed to be highly efficient, with its optimized version being approximately 2.5 times faster than LoFTR and capable of surpassing efficient sparse matching pipelines like SuperPoint + LightGlue, while also achieving higher accuracy than competitive semi-dense matchers. It processes images at a resolution of 640x480 with an optimized running time of 27.0 ms using Mixed-Precision.
- **Developed by:** ZJU3DV at Zhejiang University
- **Model type:** Image Matching
- **License:** Apache 2.0

### Model Sources

- **Repository:** https://github.com/zju3dv/efficientloftr
- **Project page:** https://zju3dv.github.io/efficientloftr/
- **Paper:** https://huggingface.co/papers/2403.04765

## Uses

Efficient LoFTR is designed for large-scale or latency-sensitive applications that require robust image matching. Its direct uses include:
- Image retrieval 
- 3D reconstruction 
- Homography estimation 
- Relative pose recovery 
- Visual localization 

### Direct Use

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched. 
The raw outputs contain the list of keypoints detected by the backbone as well as the list of matches with their corresponding 
matching scores.
```python
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

You can use the `post_process_keypoint_matching` method from the `LightGlueImageProcessor` to get the keypoints and matches in a readable format:
```python
image_sizes = [[(image.height, image.width) for image in images]]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
for i, output in enumerate(outputs):
    print("For the image pair", i)
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
        )
```

You can visualize the matches between the images by providing the original images as well as the outputs to this method:
```python
processor.visualize_keypoint_matching(images, outputs)
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2nJZQlFToCYp_iLurvcZ4.png)

## Training Details

Efficient LoFTR is trained end-to-end using a coarse-to-fine matching pipeline.

### Training Data

The model is trained on the MegaDepth dataset , a large-scale outdoor dataset.

### Training Procedure

#### Training Hyperparameters

- Optimizer: AdamW 
- Initial Learning Rate: 4×10^−3
- Batch Size: 16 
- Training Hardware: 8 NVIDIA V100 GPUs 
- Training Time: Approximately 15 hours 

#### Speeds, Sizes, Times [optional]

Efficient LoFTR demonstrates significant improvements in efficiency:
Speed: The optimized model is approximately 2.5 times faster than LoFTR. It can surpass the efficient sparse matcher LightGlue.
For 640x480 resolution image pairs on a single NVIDIA RTX 3090 GPU, the optimized model's processing time is 35.6 ms (FP32) / 27.0 ms (Mixed-Precision).
Accuracy: The method achieves higher accuracy compared to competitive semi-dense matchers and competitive accuracy compared with semi-dense matchers at a significantly higher speed.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@inproceedings{wang2024eloftr,
  title={{Efficient LoFTR}: Semi-Dense Local Feature Matching with Sparse-Like Speed},
  author={Wang, Yifan and He, Xingyi and Peng, Sida and Tan, Dongli and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2024}
}
```

## Model Card Authors

[Steven Bucaille](https://github.com/sbucaille)