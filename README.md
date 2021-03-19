# NODE for NIROM

Using a Tensorflow-based implementation of Neural ODEs (NODE) to develop non-intrusive reduced order models for CFD problems.
Numerical comparisons are made with non-intrusive reduced order models (NIROM) that use Dynamic Mode Decomposition (DMD) 
as well as a combination of linear dimension reduction using Proper Orthogonal Decomposition (POD) and latent-space 
evolution using Radial Basis Function (RBF) interpolation. 


## Description

* High-fidelity data files are available at [RDEDrive](https://rdedrive.erdc.dren.mil/url/rngn5jdnhxaizsry) (VPN required)



## Getting Started


### Dependencies

* Python 3.x
* Tensorflow 2.x
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


## Acknowledgments

* Thank you to ERDC-HPC facilities for support with valuable computational infrastructure
* Thank you to ORISE for support with appointment to the Postgraduate Research Participation Program.

Inspiration, code snippets, etc.
* [tfdiffeq](https://github.com/titu1994/tfdiffeq)
* [ML-ROM-Closures](https://github.com/Romit-Maulik/ML_ROM_Closures)
