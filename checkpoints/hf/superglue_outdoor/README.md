---
library_name: transformers
inference: false
license: other
tags:
- keypoint-matching
---

# SuperGlue

The SuperGlue model was proposed
in [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763) by Paul-Edouard Sarlin, Daniel
DeTone, Tomasz Malisiewicz and Andrew Rabinovich.

This model consists of matching two sets of interest points detected in an image. Paired with the 
[SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and 
estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following:

*This paper introduces SuperGlue, a neural network that matches two sets of local features by jointly finding correspondences 
and rejecting non-matchable points. Assignments are estimated by solving a differentiable optimal transport problem, whose costs 
are predicted by a graph neural network. We introduce a flexible context aggregation mechanism based on attention, enabling 
SuperGlue to reason about the underlying 3D scene and feature assignments jointly. Compared to traditional, hand-designed heuristics, 
our technique learns priors over geometric transformations and regularities of the 3D world through end-to-end training from image 
pairs. SuperGlue outperforms other learned approaches and achieves state-of-the-art results on the task of pose estimation in 
challenging real-world indoor and outdoor environments. The proposed method performs matching in real-time on a modern GPU and 
can be readily integrated into modern SfM or SLAM systems. The code and trained weights are publicly available at this [URL](https://github.com/magicleap/SuperGluePretrainedNetwork).*

<img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2I8QDRNoMhQCuL236CvdN.png" alt="drawing" width="500"/>

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2I8QDRNoMhQCuL236CvdN.png) -->

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperGluePretrainedNetwork).

## Demo notebook

A demo notebook showcasing inference + visualization with SuperGlue can be found [here](https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/SuperGlue_inference.ipynb).

## Model Details

### Model Description

SuperGlue is a neural network that matches two sets of local features by jointly finding correspondences and rejecting non-matchable points. 
It introduces a flexible context aggregation mechanism based on attention, enabling it to reason about the underlying 3D scene and feature 
assignments. The architecture consists of two main components: the Attentional Graph Neural Network and the Optimal Matching Layer.

<img src="https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/zZGjSWQU2na5aPFRak5kp.png" alt="drawing" width="1000"/>

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/zZGjSWQU2na5aPFRak5kp.png) -->

The Attentional Graph Neural Network uses a Keypoint Encoder to map keypoint positions and visual descriptors. 
It employs self- and cross-attention layers to create powerful representations. The Optimal Matching Layer creates a 
score matrix, augments it with dustbins, and finds the optimal partial assignment using the Sinkhorn algorithm.

- **Developed by:** MagicLeap
- **Model type:** Image Matching
- **License:** ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/magicleap/SuperGluePretrainedNetwork
- **Paper:** https://arxiv.org/pdf/1911.11763
- **Demo:** https://psarlin.com/superglue/

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

SuperGlue is designed for feature matching and pose estimation tasks in computer vision. It can be applied to a variety of multiple-view 
geometry problems and can handle challenging real-world indoor and outdoor environments. However, it may not perform well on tasks that 
require different types of visual understanding, such as object detection or image classification.


## How to Get Started with the Model


Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests
url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
im1 = Image.open(requests.get(url, stream=True).raw)
url = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
im2 = Image.open(requests.get(url, stream=True).raw)
images = [im1, im2]
processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")
inputs = processor(images, return_tensors="pt")
outputs = model(**inputs)
```

The outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding matching scores.
Due to the nature of SuperGlue, to output a dynamic number of matches, you will need to use the mask attribute to retrieve the respective information:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests
url_image_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)
images = [image_1, image_2]
processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_indoor")
model = AutoModel.from_pretrained("stevenbucaille/superglue_indoor")
inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# Get the respective image masks 
image0_mask, image1_mask = outputs_mask[0]
image0_indices = torch.nonzero(image0_mask).squeeze()
image1_indices = torch.nonzero(image1_mask).squeeze()
image0_matches = outputs.matches[0, 0][image0_indices]
image1_matches = outputs.matches[0, 1][image1_indices]
image0_matching_scores = outputs.matching_scores[0, 0][image0_indices]
image1_matching_scores = outputs.matching_scores[0, 1][image1_indices]
```

You can use the `post_process_keypoint_matching` method from the `SuperGlueImageProcessor` to get the keypoints and matches in a more readable format:
```python
image_sizes = [(image.height, image.width) for image in images]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
for i, output in enumerate(outputs):
    print("For the image pair", i)
    for keypoint0, keypoint1, matching_score in zip(output["keypoints0"], output["keypoints1"],
                                                    output["matching_scores"]):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
        )
```

From the outputs, you can visualize the matches between the two images using the following code:
```python
import matplotlib.pyplot as plt
import numpy as np
# Create side by side image
merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
merged_image[: image2.height, image1.width :] = np.array(image2) / 255.0
plt.imshow(merged_image)
plt.axis("off")
# Retrieve the keypoints and matches
output = outputs[0]
keypoints0 = output["keypoints0"]
keypoints1 = output["keypoints1"]
matching_scores = output["matching_scores"]
keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()
# Plot the matches
for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
        keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
):
    plt.plot(
        [keypoint0_x, keypoint1_x + image1.width],
        [keypoint0_y, keypoint1_y],
        color=plt.get_cmap("RdYlGn")(matching_score.item()),
        alpha=0.9,
        linewidth=0.5,
    )
    plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
    plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)
# Save the plot
plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
plt.close()
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/_2a_V28C1eEMY_Wy6sRZa.png)


## Training Details

### Training Data

SuperGlue is trained on large annotated datasets for pose estimation, enabling it to learn priors for pose estimation and reason about the 3D scene.
The training data consists of image pairs with ground truth correspondences and unmatched keypoints derived from ground truth poses and depth maps.

### Training Procedure 

SuperGlue is trained in a supervised manner using ground truth matches and unmatched keypoints. The loss function maximizes 
the negative log-likelihood of the assignment matrix, aiming to simultaneously maximize precision and recall.

#### Training Hyperparameters

- **Training regime:** fp32

#### Speeds, Sizes, Times

SuperGlue is designed to be efficient and runs in real-time on a modern GPU. A forward pass takes approximately 69 milliseconds (15 FPS) for an indoor image pair. 
The model has 12 million parameters, making it relatively compact compared to some other deep learning models.
The inference speed of SuperGlue is suitable for real-time applications and can be readily integrated into 
modern Simultaneous Localization and Mapping (SLAM) or Structure-from-Motion (SfM) systems.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@inproceedings{sarlin2020superglue,
  title={Superglue: Learning feature matching with graph neural networks},
  author={Sarlin, Paul-Edouard and DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4938--4947},
  year={2020}
}
```

## Model Card Authors

[Steven Bucaille](https://github.com/sbucaille)