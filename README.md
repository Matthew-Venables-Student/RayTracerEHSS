# RayTracerEHSS
A ray tracing algorithm that can calculate the collision cross section in a similar way to the exact hard sphere scattering method.

# Running RayTracer
RayTracer can be run using the following command:

```
./RayTracer -- (input coordinate file, either .pdb or .xyz) --orientations (default: 300) --impacts (default: 400) --projectile-radius (default 1.0 Å) --sampling (default quasi)
```
#### Inputs
RayTracer at the moment only accepts .pdb and .xyz files for proteins and small molecules respectively. Currently RayTracer is only parameterised for the standard non-metal atoms (H, C, N, O, F, S, Cl, P) and some metals (Na, K, and Fe)

#### Orientations
Orientations is the number of different orientations the input coordinates have been rotated by and averaged across. Note: the number of orientations will have a major effect on the accuracy of the calculated CCS. 
Provisionally, I'd recommend using at least 10000 orientations, (especially for big proteins), however more orientations will provide better accuracy and less spread/error.

#### Impacts
The number of rays (gas probes) used to evaluate the CCS of each orientation. Note: the number of impacts may increase calculation times depending on the number of threads used, more impacts decreases the time used to calculate each trajectory, but will increase the number of trajectories, and will have a net increase in the calculation speeds. Additionally there will be little to no improvements in accuracy with the number of impacts
Provisionally I'd recommend using between 100-500 impacts per orientation 

#### Projectile Radius 
The gas probe interaction radius that is added to the atomic radius. Most people use the following (don't ask me why) He: 1.00 Å $N_2$: 1.81 Å. Note: I have found better agreement when I used 2.2 Å for nitrogen, but I haven't extensively benchmarked this yet. Additionally RayTracer uses the VdW parameters from Sui et. al (https://doi.org/10.1021/jp910858z) that give good agreement with experimental (provisionally)

#### Sampling
There are two types of sampling used in RayTracer: "random", and "quasi".
"random"- uses a Monte Carlo method to generate the trajectories, will be pseudorandom and change across runs. This method of sampling, however, will give errors if needed.
"quasi"- uses a quasi-Monte Carlo method with the Halton sequence for both orientation and trajectories to have a more deterministic, less random, result. This method of sampling cannot give errors, but will give the spread between orientations.

# Installation
I have provided the precompiled binary of RayTracer. However for users that want to compile it,
