# 3D GAN for Microstructure Generation

## Overview

This project implements a 3D Generative Adversarial Network (GAN) for generating binary two-phase microstructures from 3D voxel data.

The input microstructures are represented as 3D binary arrays:

X ∈ {0,1}<sup>128 × 128 × 64</sup>

where each voxel represents one of two phases.

The project uses a 3D convolutional GAN consisting of:

- 3D Generator
- 3D Discriminator
- 3D transposed convolutions
- 3D convolutions
- Batch normalization
- ReLU and LeakyReLU activations
- Adam optimization
- Hinge adversarial loss
- 90% training and 10% validation split
- Latent random noise
- One-hot conditioning

---

# 1. Microstructure Representation

A microstructure describes the spatial arrangement of different material phases.

For a two-phase material, each voxel can be represented as:

X<sub>ijk</sub> ∈ {0,1}

where:

- `0` represents phase 0
- `1` represents phase 1

The complete microstructure is therefore:

X ∈ {0,1}<sup>N<sub>x</sub> × N<sub>y</sub> × N<sub>z</sub></sup>

For this project:

N<sub>x</sub> = 128

N<sub>y</sub> = 128

N<sub>z</sub> = 64

Therefore, each microstructure contains:

128 × 128 × 64 = 1,048,576

voxels.

The microstructure contains spatial information, not just the total amount of each phase.

Two structures can have the same volume fraction but completely different:

- Phase connectivity
- Interface area
- Characteristic length scale
- Tortuosity
- Anisotropy
- Transport properties
- Mechanical response

Therefore, the spatial distribution of the voxels is important.

---

# 2. Physics of the Problem

A material microstructure affects its macroscopic physical properties.

For example, consider a two-phase material with thermal conductivities:

k<sub>0</sub>

and

k<sub>1</sub>

for phase 0 and phase 1 respectively.

The effective thermal conductivity is not generally just a simple volume-weighted average.

Instead:

k<sub>eff</sub> = f(k<sub>0</sub>, k<sub>1</sub>, microstructure)

where the microstructure controls the paths through which heat can travel.

Similarly, the effective elastic properties can be represented conceptually as:

E<sub>eff</sub> = f(E<sub>0</sub>, E<sub>1</sub>, microstructure)

Fluid transport properties can similarly depend on:

K<sub>eff</sub> = f(μ, microstructure)

where:

- K<sub>eff</sub> is the effective permeability.
- μ is the fluid viscosity.

Therefore, the objective of microstructure generation is not simply to generate random binary images.

The generated structure should reproduce important statistical and physical characteristics of real microstructures.

---

# 3. Why Generate Microstructures?

Computational materials research often requires a large number of microstructures.

A conventional workflow is:

Microstructure → Physical Simulation → Material Properties

For example:

Microstructure → Finite Element Analysis → E<sub>eff</sub>

or:

Microstructure → Heat Transfer Simulation → k<sub>eff</sub>

or:

Microstructure → Fluid Flow Simulation → K<sub>eff</sub>

These simulations can become computationally expensive when thousands or millions of candidate microstructures are required.

A generative machine learning model provides another approach:

z → G(z) → Microstructure

Once trained, generating a new structure can be much cheaper than running a complete physical simulation or detailed microstructure-generation algorithm.

---

# 4. Why Deep Learning?

Traditional computational models attempt to explicitly describe the physical mechanisms that create or evolve a microstructure.

Deep learning instead learns a mapping from examples.

Given a training dataset:

D = {X<sub>1</sub>, X<sub>2</sub>, ..., X<sub>N</sub>}

the neural network attempts to learn the underlying data distribution:

p<sub>data</sub>(X)

The generator attempts to produce samples from an approximation:

p<sub>G</sub>(X) ≈ p<sub>data</sub>(X)

This is useful when the microstructure is complicated and difficult to describe analytically.

---

# 5. Why GAN Instead of a Phase-Field Model?

A phase-field model and a GAN solve different problems.

## Phase-Field Model

A phase-field model describes the evolution of a physical system using continuous fields.

A simplified phase-field equation can be written as:

∂φ/∂t = −M δF/δφ

where:

- φ is an order parameter
- M is mobility
- F is a free-energy functional

Phase-field methods can model physical processes such as:

- Phase separation
- Grain growth
- Solidification
- Precipitation
- Coarsening
- Interface evolution

The resulting microstructure is generated through a physical evolution model.

## GAN

A GAN does not explicitly solve the physical evolution equation.

Instead, it learns the statistical distribution of structures from existing examples.

Therefore:

Phase Field:

Physics → Evolution → Microstructure

GAN:

Existing Microstructures → Learning → New Microstructures

A phase-field model is appropriate when the objective is to model the physical formation or evolution of a microstructure.

A GAN is useful when the objective is to rapidly generate new structures that resemble an existing population of microstructures.

These approaches are therefore complementary rather than direct replacements for each other.

---

# 6. GAN Does Not Replace Physics

The GAN does not automatically guarantee that a generated microstructure is physically valid.

The GAN learns the distribution of the training data:

p<sub>G</sub>(X) ≈ p<sub>data</sub>(X)

It does not inherently know physical properties such as:

E, k, K, σ<sub>y</sub>

or other material properties.

Therefore, generated structures should subsequently be evaluated using physics-based methods.

The complete workflow can be:

GAN → Generated Microstructure → Physics Simulation → Effective Properties

This combines machine learning with computational physics.

---

# 7. GAN Architecture

The model consists of two neural networks:

1. Generator
2. Discriminator

The generator creates a microstructure:

G(z,c) = X<sub>fake</sub>

where:

- z is a random latent vector
- c is the condition
- X<sub>fake</sub> is the generated microstructure

The discriminator attempts to distinguish real and generated microstructures.

---

# 8. Generator

The generator receives:

z ∈ R<sup>128</sup>

and a condition:

c ∈ R<sup>2</sup>

The combined input is:

[z,c] ∈ R<sup>130</sup>

The first fully connected layer maps this representation to:

512 × 4 × 4 × 2

The tensor is then reshaped into:

512 × 4 × 4 × 2

3D transposed convolutions progressively increase the spatial resolution:

4 × 4 × 2 → 8 × 8 × 4

8 × 8 × 4 → 16 × 16 × 8

16 × 16 × 8 → 32 × 32 × 16

32 × 32 × 16 → 64 × 64 × 32

64 × 64 × 32 → 128 × 128 × 64

The final output is:

X<sub>fake</sub> ∈ [−1,1]<sup>128 × 128 × 64</sup>

because the final layer uses the hyperbolic tangent activation:

tanh(x)

---

# 9. Why 3D Convolution?

A microstructure is inherently three-dimensional.

Using 2D convolutions independently on individual slices would lose correlations between neighboring slices.

A 3D convolution operates on:

x, y, z

simultaneously.

A simplified 3D convolution can be represented as:

Y(i,j,k) = Σ W(a,b,c) X(i+a,j+b,k+c)

Therefore, the network can learn three-dimensional spatial features such as:

- Phase clusters
- Interfaces
- Connectivity
- Pore morphology
- Three-dimensional anisotropy
- Spatial correlations

---

# 10. Discriminator

The discriminator receives a 3D microstructure:

X ∈ R<sup>128 × 128 × 64</sup>

and progressively reduces its spatial resolution:

128 × 128 × 64 → 64 × 64 × 32

64 × 64 × 32 → 32 × 32 × 16

32 × 32 × 16 → 16 × 16 × 8

16 × 16 × 8 → 8 × 8 × 4

8 × 8 × 4 → 4 × 4 × 2

The resulting features are flattened and passed through a linear layer.

The discriminator outputs one scalar:

D(X) ∈ R

---

# 11. Adversarial Learning

The generator and discriminator are trained against each other.

The discriminator attempts to distinguish:

X<sub>real</sub>

from:

X<sub>fake</sub>

The generator attempts to produce structures that the discriminator considers realistic.

Conceptually:

G ↔ D

As training progresses:

- D learns features that distinguish real and generated structures.
- G learns features that make generated structures more similar to the training distribution.

---

# 12. Hinge Loss

This implementation uses hinge adversarial loss.

The discriminator loss is:

L<sub>D</sub> = E[max(0, 1 − D(x))] + E[max(0, 1 + D(G(z)))]

The generator loss is:

L<sub>G</sub> = −E[D(G(z))]

The discriminator does not use a sigmoid output because the hinge formulation works directly with the raw discriminator score.

---

# 13. Why Hinge Loss?

Hinge loss is commonly used in modern GAN architectures.

For real samples:

L<sub>real</sub> = max(0, 1 − D(x))

For generated samples:

L<sub>fake</sub> = max(0, 1 + D(G(z)))

The discriminator is encouraged toward:

D(x) > 1

for real data and:

D(G(z)) < −1

for generated data.

The generator attempts to increase:

D(G(z))

so that generated structures are considered more realistic.

---

# 14. Optimization

The model uses the Adam optimizer.

For a parameter θ and gradient:

g<sub>t</sub> = ∇<sub>θ</sub>L<sub>t</sub>

Adam maintains first and second moment estimates:

m<sub>t</sub> = β<sub>1</sub>m<sub>t−1</sub> + (1−β<sub>1</sub>)g<sub>t</sub>

v<sub>t</sub> = β<sub>2</sub>v<sub>t−1</sub> + (1−β<sub>2</sub>)g<sub>t</sub><sup>2</sup>

followed by bias correction and parameter updates.

The implementation uses:

α = 2 × 10<sup>−4</sup>

β<sub>1</sub> = 0.5

β<sub>2</sub> = 0.999

Adam is used because GAN training involves two simultaneously changing neural networks and can be difficult to optimize using ordinary gradient descent.

---

# 15. Data Processing

The original data contains binary values:

X ∈ {0,1}

The generator uses a final tanh activation, producing:

X<sub>generated</sub> ∈ [−1,1]

Therefore, real data is transformed using:

X′ = 2X − 1

giving:

0 → −1

1 → +1

This puts the real and generated data in the same numerical range.

---

# 16. Dataset Loading

The HDF5 files contain multiple microstructures.

Each HDF5 key corresponds to one 3D volume.

Instead of loading every microstructure into RAM, the dataset stores an index:

```text
(file path, HDF5 key)



Future Plan

Once the GAN generates new 3D microstructures, the generated microstructures will be passed to CH/FEM-based simulations to evaluate their effective physical properties.

1. Thermal Conductivity

The generated microstructure will be used to solve the heat-conduction problem:

κ∇²u = 0

→ Obtain effective thermal conductivity κ<sub>eff</sub>.

2. Linear Elasticity

The microstructure will be analyzed under mechanical loading using:

∇⋅(C∇u) = 0

→ Obtain effective elastic properties such as Young's modulus and stiffness.

3. Permeability

Fluid flow through the generated porous microstructure will be modeled using:

μ∇²u − ∇p + b = 0

∇⋅u − τp = 0

→ Obtain effective permeability.

4. Thermal Expansion

Thermo-mechanical behavior will be evaluated using:

∇⋅(C∇u − CαΔτ) = 0

→ Obtain effective thermal expansion behavior.

5. Overall Workflow

Training Dataset → 3D GAN → Generated Microstructure → CH/FEM Simulation → Effective Material Properties

The generated structures can then be compared with the original dataset based on both microstructural statistics and physical properties.
