# ericksen_leslie_x
FEniCSx implementation of numerical methods for the Ericksen--Leslie equations for nematic liquid crystal flow. 

## Governing equations
The governing equations in a general form can be stated as:

$$
\begin{aligned}
\partial_t  v + ( v \cdot \nabla)  v  + \nabla  p + \nabla \cdot T^E  - \nabla \cdot  T^L = 0,
\\
\nabla \cdot  v = 0,
\\
d \times   (\partial_t  d + ( v \cdot \nabla)  d - {(\nabla  v)}_{skw}  d +  \lambda {(\nabla  v)}_{sym}  d - \Delta  d   ) =0,
\\
\vert d \vert^2 = 1.
\end{aligned}
$$

Depending on the choice of complexity for the different stress tensors, we consider the following submodels.

### Submodel 1

$$
\begin{aligned}
\partial_t  v + ( v \cdot \nabla)  v  + \nabla  p +  \nabla \cdot [\nabla  d]^T   ( I -  d \otimes  d   ) \Delta d    - \nabla \cdot  T^L = 0,
\\
\nabla \cdot  v = 0,
\\
\partial_t  d +   ( I -  d \otimes  d   ) [ ( v \cdot \nabla)  d - {(\nabla  v)}_{skw}  d +   ( \lambda {(\nabla  v)}_{sym}  d - \Delta  d   ) ] =0,
\\
\vert d \vert^2 = 1.
\end{aligned}
$$

### Submodel 2

$$
\begin{aligned}
\partial_t  v + ( v \cdot \nabla)  v  + \nabla  p  + \nabla \cdot [\nabla  d]^T   ( I -  d \otimes  d   ) \Delta d  - \nu \Delta v = 0,
\\
\nabla \cdot  v = 0,
\\
\partial_t  d +   ( I -  d \otimes  d   ) [ ( v \cdot \nabla)  d - \Delta  d   ] =0,
\\
\vert d \vert^2 = 1.
\end{aligned}
$$


## Numerical Methods

Currently the following numerical schemes are available.
- linear Continuous Galerkin (CG) scheme fulfilling a discrete energy law
    - with mass-lumping
        - with a projection step (LhP)
        - without projection step (Lh)
    - with the standard $L^2$ inner product
        - with a projection step (LL2P)
        - without projection step (LL2)
- linear Discontinuous Galerkin (DG) scheme fulfilling a discrete energy law
    - with a projection step (lpdg)
    - without projection step (ldg)
- a linearized, decoupled fixed point solver to approximate a fully implicit Continuous Galerkin (CG) scheme (FPhD)
    - with mass-lumping, --> fulfills the unit-norm constraint exactly at every node of the mesh

All of the above methods use P2-P1-Taylor-Hood Finite Elements for velocity and space and CG1 elements for the director field and its discrete Laplacian.


## Getting Started and Usage

All arguments to run simulations are given via the command line input. To see the options run the following command in the package directory:

```
python -m sim -h
```

Another usage example with several arguments:

```
python -m sim -m "LhP" -e spiral -cp -vtx -dh 20 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-experiment" -T 0.05
```

Presets for several simulations --- usually in the context of publications --- are given in the folder 'sim/sim_presets/' usually in the form of a bash or python file. Examples for usage:

```
python -m sim.sim_presets.projection-spiral
```
and
```
sim/sim_presets/spiral.sh
```

## Requirements

All requirements can be found in the file requirements.txt and can be installed via pip by

```
pip install -r requirements.txt
```

or via conda by

```
conda create --name my-env-name --file requirements.txt -c conda-forge
```

## Notes

- Experiments of the class 'spiral' need a predefined mesh in .msh or .xdmf format in the folder 'input/meshes'.
- Results are automatically named and saved in the folder 'output/'

## References to relevant publications

List references with links to publications this code was used for:
- The decoupled fixed point solver is based on [[1]](#1).

<a id="1">[1]</a> 
Lasarzik, R., Reiter, M.E.V. Analysis and Numerical Approximation of Energy-Variational Solutions to the Ericksen–Leslie Equations. Acta Appl Math 184, 11 (2023). https://doi.org/10.1007/s10440-023-00563-9


## Authors

* **Maximilian E. V. Reiter**, https://orcid.org/0000-0001-9137-7978

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
