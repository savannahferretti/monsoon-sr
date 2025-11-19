Data-Driven Discovery of Thermodynamic Controls on South Asian Monsoon Precipitation
------------

Savannah L. Ferretti<sup>1</sup>, Tom Beucler<sup>2</sup>, Michael S. Pritchard<sup>3</sup>, Sara Shamekh<sup>4</sup>, Fiaz Ahmed<sup>5</sup>, & Jane W. Baldwin<sup>1,6</sup>

<sup>1</sup>Department of Earth System Science, University of California Irvine, Irvine, CA, USA  
<sup>2</sup>Faculty of Geosciences and Environment, University of Lausanne, Lausanne, VD, CH  
<sup>3</sup>NVIDIA Corporation, Santa Clara, CA, USA  
<sup>4</sup>Courant Institute for Mathematical Science, New York University, New York, NY, USA  
<sup>5</sup>Department of Atmopsheric and Oceanic Sciences, University of California Los Angeles, Los Angeles, CA, USA  
<sup>6</sup>Lamont-Doherty Earth Observatory, Palisades, NY, USA  

**Status:** This manuscript is currently in preparation. We welcome any comments, questions, or suggestions. Please email your feedback to Savannah Ferretti (savannah.ferretti@uci.edu).

**Key Points**:
- Point 1
- Point 2
- Point 3

**Abstract**: Insert abstract text here.

Project Organization
------------
```
├── LICENSE.md         <- License for code
│
├── README.md          <- Top-level information on this code base/manuscript
│
├── data/
│   ├── raw/           <- Original ERA5 and IMERG V06 data
│   ├── interim/       <- Intermediate data that has been transformed
│   ├── splits/        <- Training, validation, and test sets
│   └── results/       <- Model predictions (and skill metrics)
│
├── figs/              <- Generated figures/graphics 
│
├── models/            
│   ├── pod/           <- Saved POD models
│   ├── nn/            <- Saved NN models
│   └── sr/            <- Saved PySR models
│
├── notebooks/         <- Jupyter notebooks for data analysis and visualizations
│
├── scripts/             
│   ├── data/          <- Scripts for downloading, and calculating input/target terms, and splitting data
│   ├── pod/           <- Scripts for training/evaluating the POD (baseline) model
│   ├── nn/            <- Scripts for training/evaluating the full-profile and kernel NNs
│   └── sr/            <- Scripts for running PySR and summarizing discovered equations 
│
└── environment.yml    <- File for reproducing the analysis environment
```

Acknowledgements
-------

The analysis for this work has been performed on NERSC's [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was primarily funded by the DOE's [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research)
Program, with additional support from [LEAP NSF-STC](https://leap.columbia.edu/).

--------
<p><small>This template is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
