---
title: 'EdelweissFE: A light-weight, platform-independent, parallel finite element framework'
tags:
  - Python
  - Cython
  - finite elements
  - nonlinear mechanics
  - constitutive models
  - geomechanics
authors:
  - name: Matthias Neuner
    orcid: 0000-0003-2221-8089
    affiliation: "1, 2, 3"
    corresponding: true
  - name: Alexander Dummer
    orcid: 0000-0002-1998-1705
    affiliation: "1, 4"
  - name: Daniel Reitmair
    orcid: 0009-0002-2748-5442
    affiliation: 1
  - name: Magdalena Schreter-Fleischhacker
    orcid: 0000-0003-3888-4086
    affiliation: "1, 5"
  - name: Paul Hofer
    orcid: 0009-0001-6589-7993
    affiliation: 1
affiliations:
  - name: University of Innsbruck, Austria
    index: 1
  - name: Stanford University, USA
    index: 2
  - name: BOKU University, Austria
    index: 3
  - name: University of Colorado Boulder, USA
    index: 4
  - name: Technical University Munich, Germany
    index: 5
date: 27 May 2026
bibliography: paper.bib
---

# Summary

EdelweissFE is a light-weight, platform-independent, parallel finite element framework written in Python and Cython.
It is designed as a flexible development and learning environment for constitutive models and finite elements,
with a focus on nonlinear solid mechanics, geomechanics, and coupled problems.
EdelweissFE combines the readability and ease of use of Python for non performance-critical routines with
Cython-accelerated kernels for computationally intensive operations, achieving a balance between
accessibility and computational efficiency for problems up to medium size (~$10^5$ degrees of freedom).

By default, EdelweissFE interfaces with the Marmot library [@Marmot] for finite element and constitutive model
formulations, enabling access to a wide range of advanced material models including gradient-enhanced
damage-plasticity formulations and micropolar continuum models.

# Statement of need

The development of novel constitutive models and finite element formulations requires a flexible
computational environment that allows rapid prototyping, debugging, and validation.
While mature, MPI-parallelized finite element frameworks such as MOOSE [@moose], FEniCS [@fenics],
and deal.II [@dealII] offer high performance for production-scale simulations, their complexity
can hinder the iterative development process of new numerical methods and material models.
Commercial codes such as Abaqus, while widely used, lack the transparency and extensibility
needed for fundamental research.

EdelweissFE fills this gap by providing:

- A modular architecture that makes it straightforward to implement and test new element formulations,
  constitutive models, solvers, and solution techniques.
- A Python-based input system that is easy to understand and modify.
- Shared-memory parallelization for efficient computation on workstations.
- Multiple output formats (Paraview/VTK, Ensight, CSV, matplotlib) for flexible post-processing.
- Interfaces to direct solvers (e.g., Pardiso via MKL) and iterative solvers (e.g., AMGCL).
- Special techniques such as indirect displacement control, which are difficult to implement
  in large-scale MPI-parallelized codes.

The typical workflow involves developing and validating constitutive models in EdelweissFE before
deploying them in production frameworks for large-scale HPC simulations.
EdelweissFE has been used in several peer-reviewed publications on topics including
borehole breakout mechanics [@Neuner2022borehole], gradient-enhanced micropolar continuum
models [@Neuner2022unified], creep-induced failure of concrete [@Dummer2022],
and regularization techniques for damage-plasticity models [@Schreter2018; @Neuner2020].

# Key features

- **Python + Cython**: Non performance-critical code is written in Python for readability;
  performance-critical assembly and constitutive evaluation routines use Cython.
- **Parallelization**: Shared-memory parallelization via domain decomposition.
- **Modularity**: Plugin-based architecture for elements, materials, solvers, step actions, and output modules.
- **Marmot integration**: Seamless interface to the Marmot C++ library for advanced constitutive models.
- **Comprehensive output**: Support for Paraview (VTK/VTU), Ensight Gold, CSV, and matplotlib.

# Acknowledgements

The development of EdelweissFE was supported by the University of Innsbruck.
The authors thank all contributors who have participated in the development of the framework,
in particular Konstantin Basche, Thomas Mader, and Johannes Thiel.

# References
