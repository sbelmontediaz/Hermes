Hermes

Hermes is a Python package for performing direct DM–time domain searches for fast radio transients. The backbone of the pipeline is a Mask R-CNN, an image segmentation deep learning model. The package provides tools to construct physically motivated dedispersion plans and DM–time tiling strategies suitable for deep learning pipelines that operate directly on DM–time search windows, rather than on pre-selected candidates.

Hermes was developed in support of the paper:

Enhancing Fast Radio Transient Detection with Mask R-CNN Image Segmentation
Belmont-Díaz et al. (submitted)

Installation

Clone the repository:

git clone https://github.com/sbelmontediaz/Hermes.git
cd Hermes


Install the dependencies (example):

pip install -r requirements.txt

Basic Usage

To run the full pipeline, modify the config.yaml file to specify:

- the trial widths and DM ranges,

- the path to the trained model weights,

- the batch size appropriate for the available hardware.

The end-to-end pipeline is implemented in main.py. To view the available command-line options, run:

python main.py -h


The pipeline operates directly on filterbank files.

Dedispersion Planning

The dedispersion planning tool can be used independently. To generate a dedispersion plan, run:

python ddplan.py output.txt


Ensure that the config.yaml file is updated with the correct observing setup (e.g. top and bottom frequencies, number of channels, time resolution).
