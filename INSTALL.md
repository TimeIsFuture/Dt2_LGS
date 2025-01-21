## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.4
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, optional, needed by demo and visualization

### Build Dt2_LGS from Source

gcc & g++ ≥ 5 are required. [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
# install it from a local clone:
git clone https://github.com/TimeIsFuture/detectron2_LGS.git
python -m pip install -e detectron2_LGS
```

To __rebuild__ Dt2_LGS that's built from a local clone, use `rm -rf build/ **/*.so` to clean the
old build first. You often need to rebuild Dt2_LGS after reinstalling PyTorch.
