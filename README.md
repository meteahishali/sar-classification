PolSAR Classification using Convolutional Neural Networks
=============================

This repository includes a Tensorflow implentation of the method in [Classification of Polarimetric SAR Images using Compact Convolutional Neural Networks](https://www.tandfonline.com/doi/full/10.1080/15481603.2020.1853948). Note that the provided implementation is not identical to the originally proposed approach. On the other hand, CNN arhitecture, electromagnetic channels and sliding window classification procedures are the same with the original C++ implementation.


Software environment:
```
conda create -n tensorflow_2.1.0 python=3.7

conda activate tensorflow_2.1.0

conda install tensorflow-gpu=2.1.0 scikit-learn

pip install matplotlib

pip install h5py==2.10.0
```


The datasets can be originally found in the following sources:

AIRSAR San Francisco PolSAR Data at L-band (sfbay_l): https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/.

RADARSAT-2 San Francisco PolSAR Data at C-band (sfbay_c): https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/.

AIRSAR Flevoland PolSAR Data at L-band (flevo_l): https://earth.esa.int/web/polsarpro/datasources/sample-datasets

RADARSAT-2 Flevoland PolSAR Data at C-band (flevo_c): https://www.esa.int/ESA_Multimedia/Images/2009/04/Radarsat-2_image_of_Flevoland_in_the_Netherlands.

In this repository, we have provided the pre-processed data: we first extract the diagonal elements of the average polarimetric covariance matrix, coherency matrix, and total scattering power, then apply dB conversion and finally linear-scaling to have the input channels in the range of [-1, 1]. Run the provided ```bin2pkl.py``` script to organize the data for the CNN. You can specify the number of channels ```-c``` i.e., ```3, 4, 6``` and dataset ```sfbay_l, sfbay_c, flevo_l``` or ```flevo_c```. Specify also the size of sliding window ```-p``` (patch size). For example,
```
python bin2pkl.py -c 6 -p 7 --dataset sfbay_l -W 1024 -H 900

python bin2pkl.py -c 6 -p 5 --dataset sfbay_c -W 1426 -H 1876

python bin2pkl.py -c 6 -p 5 --dataset flevo_l -W 1024 -H 750

python bin2pkl.py -c 6 -p 5 --dataset flevo_c -W 1639 -H 2393
```

Classification:

```
python classify.py -c 6 -p 7 --dataset sfbay_l

python classify.py -c 6 -p 7 --dataset sfbay_c

python classify.py -c 6 -p 7 --dataset flevo_l

python classify.py -c 6 -p 7 --dataset flevo_c
```

## Citation

If you use the provided implementation in this repository, please cite the following paper:

```
@article{ahishali,
author = {Mete Ahishali and Serkan Kiranyaz and Turker Ince and Moncef Gabbouj},
title = {Classification of polarimetric SAR images using compact convolutional neural networks},
journal = {GIScience \& Remote Sensing},
volume = {58},
number = {1},
pages = {28-47},
year  = {2021},
publisher = {Taylor & Francis},
doi = {10.1080/15481603.2020.1853948},
}
```


