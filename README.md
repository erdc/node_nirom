# NODE for NIROM

Using a Tensorflow-based implementation of Neural ODEs (NODE) to develop non-intrusive reduced order models for CFD problems.
Numerical comparisons are made with non-intrusive reduced order models (NIROM) that use Dynamic Mode Decomposition (DMD) 
as well as a combination of linear dimension reduction using Proper Orthogonal Decomposition (POD) and latent-space 
evolution using Radial Basis Function (RBF) interpolation. 

For details please refer to
```
S. Dutta, P. Rivera-casillas, and M. W. Farthing, “Neural Ordinary Differential Equations for Data-Driven Reduced Order Modeling of Environmental Hydrodynamics,” in Proceedings of the AAAI 2021 Spring Symposium on Combining Artificial Intelligence and Machine Learning with Physical Sciences, 2021. [Link](https://sites.google.com/view/aaai-mlps/proceedings?authuser=0)
```


## Getting Started


### Dependencies

* Python 3.x
* Tensorflow 2.x, 1.15.x
* tfdiffeq


### Executing program

* NODE scripts, available inside the notebooks directory, can be invoked with various user-specified configuration options to test different NN models 
* DMD and PODRBF notebooks are also available inside the notebooks directory.
* High-fidelity snapshot data files are available at [HFM data](https://drive.google.com/drive/folders/1uRgkcA36tRZQqdfUDunhVNaqxa1Cceqd?usp=sharing). These should be placed in the <node\_nirom/data/> directory.
* Some pre-trained ROM model files are available at [NIROM models](https://drive.google.com/drive/folders/19DEWdoS7Fkh-Cwe7Lbq6pdTdE290gYSS?usp=sharing). The DMD and PODRBF trained models should be placed in the <node\_nirom/data/> directory, and the NODE models should be placed inside the corresponding subdirectory of <node\_nirom/best\_models>.

## Authors

* **Sourav Dutta** - *Sourav.Dutta@erdc.dren.mil* - ERDC-CHL
* **Matthew Farthing** - *Matthew.W.Farthing@erdc.dren.mil* - ERDC-CHL
* **Peter Rivera-Casillas** - *Peter.G.Rivera-Casillas@erdc.dren.mil* - ERDC-CHL 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Reference

If you found this library useful in your research, please consider citing
```
@inproceedings{dutta2021aaai,
title={Neural Ordinary Differential Equations for Data-Driven Reduced Order Modeling of Environmental Hydrodynamics},
author={Dutta, Sourav and Rivera-Casillas, Peter and Farthing, Matthew W.},
booktitle={Proceedings of the AAAI 2021 Spring Symposium on Combining Artificial Intelligence and Machine Learning with Physical Sciences},
url={https://sites.google.com/view/aaai-mlps/proceedings?authuser=0},
year={2021},
publisher={CEUR-WS},
address={Stanford, CA, USA, March 22nd to 24th, 2021},
}
```


## Acknowledgments

* Thank you to ERDC-HPC facilities for support with valuable computational infrastructure
* Thank you to ORISE for support with appointment to the Postgraduate Research Participation Program.

Inspiration, code snippets, etc.
* [tfdiffeq](https://github.com/titu1994/tfdiffeq)
* [ML-ROM-Closures](https://github.com/Romit-Maulik/ML_ROM_Closures)
