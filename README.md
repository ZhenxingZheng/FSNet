# Collaborative and Hierarchical Feature Selection Network for Action Recognition

**Abstract:** The feature pyramid as a valid representation has been widely used in many visual tasks, such as fine-grained image classification, instance segmentation, and object detection, achieving promising performance. Although many algorithms exploit different-scale features to construct the feature pyramid, they usually treat them equally and do not make an in-depth exploration of the inherent complementary advantages of different-level features. In this paper, to learn a pyramid feature with the robust representational ability for action recognition, we propose a novel collaborative and multi-level Feature Selection Network (FSNet) that applies feature selection and aggregation on multi-level features according to action context. Unlike previous works that learn the appearance pattern of frames by enhancing spatial encoding, the proposed network consists of two core modules, the position selection module and channel selection module, to adaptively integrate multi-level features into a new informative feature from both position and channel dimensions. The position selection module pools the vectors at the same spatial location across multi-level features with position-wise attention. Similarly, the channel selection module selectively aggregates the channel maps at the same channel location across multi-level features with channel-wise attention. Position-wise information with different receptive fields and channel-wise information with different pattern-specific responses are emphasized depending on their correlations to actions and thereafter are fused as a new informative feature for action recognition. The proposed FSNet can be integrated into 2D and 3D backbone networks flexibly and extensive experiments are conducted on three benchmark action datasets, Kinetics, UCF101, and HMDB51. Experimental results show that FSNet is practical and can be collaboratively learned to boost the representational power of existing networks. FSNet achieves superior performance against most top-tier models on Kinetics and all methods on UCF101 and HMDB51 respectively.


## Requirements

PyTorch 0.4.1

opencv-python


## Datasets

The datasets and splits can be downloaded from

[Kinetics-400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)

[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

## Models
The pretrained models on Kinetics can be downloaded from [Baidu Yun](https://pan.baidu.com/s/1VhXSlWOYtZeku1LzcQq1gg), code:o4ib.


## Comparison with state-of-the-art methods

| Model                    | HMDB51 | UCF101 |
| ------------------------ | ------ | ------ |
| TEA-ResNet50[1]          | 73.3   | 96.9   |
| Dense Dilated Network[2] | 74.5   | 96.9   |
| MARS+RGB[3]              | 79.5   | 97.6   |
| Two-Stream I3D[4]        | 80.2   | 97.9   |
| PoTion + I3D[5]          | 80.9   | 98.2   |
| FSNet(RGB, ours)         | 81.4   | 98.3   |
| FSNet(RGB)+I3D(Flow)     | 84.8   | 98.9   |


## References

[1]Y. Li, B. Ji, X. Shi, J. Zhang, B. Kang, and L. Wang, “Tea: Temporal excitation and aggregation for action recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2020, pp. 909–918.

[2]B. Xu, H. Ye, Y. Zheng, H. Wang, T. Luwang, and Y. Jiang, “Dense dilated network for video action recognition,” IEEE Trans. Image Process., vol. 28, no. 10, pp. 4941–4953, 2019.

[3] N. Crasto, P. Weinzaepfel, K. Alahari, and C. Schmid, “Mars: Motionaugmented rgb stream for action recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 7882–7891.

[4]J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 4724–4733.

[5] V. Choutas, P. Weinzaepfel, J. Revaud, and C. Schmid, “PoTion: Pose MoTion representation for action recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018, pp. 7024–7033.


## Comming Soon
More details will be released.
