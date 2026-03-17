---
library_name: transformers
license: other
pipeline_tag: keypoint-detection
tags:
- keypoint-matching
---

# LightGlue

The LightGlue model was proposed
in [LightGlue: Local Feature Matching at Light Speed](http://arxiv.org/abs/2306.13643) by Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys.

This model consists of matching two sets of interest points detected in an image. Paired with the 
[SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and 
estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following :
We introduce LightGlue, a deep neural network that learns to match local features across images. We revisit multiple 
design decisions of SuperGlue, the state of the art in sparse matching, and derive simple but effective improvements. 
Cumulatively, they make LightGlue more efficient – in terms of both memory and computation, more accurate, and much 
easier to train. One key property is that LightGlue is adaptive to the difficulty of the problem: the inference is 
much faster on image pairs that are intuitively easy to match, for example because of a larger visual overlap or 
limited appearance change. This opens up exciting prospects for deploying deep matchers in latency-sensitive 
applications like 3D reconstruction. The code and trained models are publicly available at [github.com/cvg/LightGlue](https://github.com/cvg/LightGlue).


<img src="https://raw.githubusercontent.com/cvg/LightGlue/main/assets/easy_hard.jpg" alt="drawing" width="800"/>

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/cvg/LightGlue).

## Demo notebook

A demo notebook showcasing inference + visualization with LightGlue can be found [TBD]().


## Model Details

### Model Description

LightGlue is a neural network that matches two sets of local features by jointly finding correspondences and rejecting non-matchable points.
Building on the success of SuperGlue, this model has the ability to introspect the confidence of its own predictions. It adapts the amount of
computation to the difficulty of each image pair to match. Both its depth and width are adaptive : 
1. the inference can stop at an early layer if all predictions are ready
2. points that are deemed not matchable are discarded early from further steps.
The resulting model, LightGlue, is finally faster, more accurate, and easier to train than the long-unrivaled SuperGlue.

<img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/ILpGyHuWwK2M9Bz0LmZLh.png" alt="drawing" width="1000"/>

- **Developed by:** ETH Zurich - Computer Vision and Geometry Lab 
- **Model type:** Image Matching
- **License:** ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY (implied by the use of SuperPoint as its keypoint detector)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/cvg/LightGlue
- **Paper:** http://arxiv.org/abs/2306.13643
- **Demo:** https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

LightGlue is designed for feature matching and pose estimation tasks in computer vision. It can be applied to a variety of multiple-view 
geometry problems and can handle challenging real-world indoor and outdoor environments. However, it may not perform well on tasks that 
require different types of visual understanding, such as object detection or image classification.

## How to Get Started with the Model

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched. 
The raw outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding 
matching scores.
```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

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
processor.plot_keypoint_matching(images, outputs)
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/duPp09ty8NRZlMZS18ccP.png)


## Training Details

LightGlue is trained on large annotated datasets for pose estimation, enabling it to learn priors for pose estimation and reason about the 3D scene.
The training data consists of image pairs with ground truth correspondences and unmatched keypoints derived from ground truth poses and depth maps.

LightGlue follows the supervised training setup of SuperGlue. It is first pre-trained with synthetic homographies sampled from 1M images. 
Such augmentations provide full and noise-free supervision but require careful tuning. LightGlue is then fine-tuned with the MegaDepth dataset, 
which includes 1M crowd-sourced images depicting 196 tourism landmarks, with camera calibration and poses recovered by SfM and 
dense depth by multi-view stereo.

#### Training Hyperparameters

- **Training regime:** fp32

#### Speeds, Sizes, Times

LightGlue is designed to be efficient and runs in real-time on a modern GPU. A forward pass takes approximately 44 milliseconds (22 FPS) for an image pair. 
The model has 13.7 million parameters, making it relatively compact compared to some other deep learning models.
The inference speed of LightGlue is suitable for real-time applications and can be readily integrated into 
modern Simultaneous Localization and Mapping (SLAM) or Structure-from-Motion (SfM) systems.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}
```

## Model Card Authors

[Steven Bucaille](https://github.com/sbucaille)