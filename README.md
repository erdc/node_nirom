# NODE for NIROM

Using a Tensorflow-based implementation of Neural ODEs (NODE) to develop non-intrusive reduced order models for CFD problems.
Numerical comparisons are made with non-intrusive reduced order models (NIROM) that use Dynamic Mode Decomposition (DMD) 
as well as a combination of linear dimension reduction using Proper Orthogonal Decomposition (POD) and latent-space 
evolution using Radial Basis Function (RBF) interpolation. 


## Description

* High-fidelity snapshots data files and trained model files are available at [RDEDrive](https://rdedrive.erdc.dren.mil/url/rngn5jdnhxaizsry) (VPN required)


## Getting Started


### Dependencies

* Python 3.x
* Tensorflow 2.x, 1.15.x
* tfdiffeq


### Executing program

* NODE scripts, available inside the notebooks directory, can be invoked with various user-specified configuration options to test different NN models 
* DMD and PODRBF notebooks are also available inside the notebooks directory.


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
