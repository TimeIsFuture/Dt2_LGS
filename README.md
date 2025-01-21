<img src=".github/Dt2_LGS-Logo-Horz.svg" width="300" >

<div align="center">
  <img src="./SAandLGS.png"/>
</div>

(a) The self-attention enables the pixel embedding interaction. The softmax correlation is computed between pixels. 
(b) Our LGS module learns modeling the local semantic aggregation and global semantic interaction. 
LGS contains a group convolution and an Efficient Global Semantic Attention (EGSA). 
EGSA formulates a general model for the global semantic interaction. 
The linear semantic correlation is computed between channels. 
LGS has the linear memory overhead and computation cost in term of the spatial resolution.
The codes are built on the open-source toolkit [detectron2](https://github.com/facebookresearch/detectron2/tree/main).

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),

## Model Zoo

We provide a large set of baseline results for download in the [Dt2_LGS Model Zoo](MODEL_ZOO.md).

## License

Dt2_LGS is released under the [Apache 2.0 license](LICENSE).

## Citing Dt2_LGS

If you use Dt2_LGS in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

@article{Li2024LGSNetLG,
  author={Li, Yang and Jiao, Licheng and Liu, Xu and Liu, Fang and Li, Lingling and Chen, Puhua},
  journal={IEEE Transactions on Multimedia}, 
  title={LGSNet: Local-Global Semantics Learning Object Detection}, 
  year={2024},
  pages={1-12},
  doi={10.1109/TMM.2024.3521850}}
```
