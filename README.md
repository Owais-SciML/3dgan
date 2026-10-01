# 3D GAN for Microstructure Generation

## Overview

This project implements a 3D Generative Adversarial Network (GAN) for generating binary two-phase microstructures from 3D voxel data.

The input microstructures are represented as 3D binary arrays:

$$
X \in \{0,1\}^{128 \times 128 \times 64}
$$

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

$$
X_{ijk} \in \{0,1\}
$$

where:

- $0$ represents phase 0
- $1$ represents phase 1

The complete microstructure is therefore:

$$
X \in \{0,1\}^{N_x \times N_y \times N_z}
$$

For this project:

$$
N_x = 128,\qquad
N_y = 128,\qquad
N_z = 64
$$

Therefore, each microstructure contains:

$$
128 \times 128 \times 64 = 1,048,576
$$

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

$$
k_0
$$

and

$$
k_1
$$

for phase 0 and phase 1 respectively.

The effective thermal conductivity is not generally just a simple volume-weighted average.

Instead:

$$
k_{\mathrm{eff}}
=
f(k_0,k_1,\text{microstructure})
$$

where the microstructure controls the paths through which heat can travel.

Similarly, the effective elastic properties can be represented conceptually as:

$$
E_{\mathrm{eff}}
=
f(E_0,E_1,\text{microstructure})
$$

Fluid transport properties can similarly depend on:

$$
K_{\mathrm{eff}}
=
f(\mu,\text{microstructure})
$$

where:

- $K_{\mathrm{eff}}$ is effective permeability
- $\mu$ is fluid viscosity

Therefore, the objective of microstructure generation is not simply to generate random binary images.

The generated structure should reproduce important statistical and physical characteristics of real microstructures.

---

# 3. Why Generate Microstructures?

Computational materials research often requires a large number of microstructures.

A conventional workflow is:

$$
\text{Microstructure}
\rightarrow
\text{Physical Simulation}
\rightarrow
\text{Material Properties}
$$

For example:

$$
\text{Microstructure}
\rightarrow
\text{Finite Element Analysis}
\rightarrow
E_{\mathrm{eff}}
$$

or:

$$
\text{Microstructure}
\rightarrow
\text{Heat Transfer Simulation}
\rightarrow
k_{\mathrm{eff}}
$$

or:

$$
\text{Microstructure}
\rightarrow
\text{Fluid Flow Simulation}
\rightarrow
K_{\mathrm{eff}}
$$

These simulations can become computationally expensive when thousands or millions of candidate microstructures are required.

A generative machine learning model provides another approach:

$$
z
\rightarrow
G(z)
\rightarrow
\text{Microstructure}
$$

Once trained, generating a new structure can be much cheaper than running a complete physical simulation or detailed microstructure-generation algorithm.

---

# 4. Why Deep Learning?

Traditional computational models attempt to explicitly describe the physical mechanisms that create or evolve a microstructure.

Deep learning instead learns a mapping from examples.

Given a training dataset:

$$
\mathcal{D}
=
\{X_1,X_2,\ldots,X_N\}
$$

the neural network attempts to learn the underlying data distribution:

$$
p_{\mathrm{data}}(X)
$$

The generator attempts to produce samples from an approximation:

$$
p_G(X)
\approx
p_{\mathrm{data}}(X)
$$

This is useful when the microstructure is complicated and difficult to describe analytically.

---

# 5. Why GAN Instead of a Phase-Field Model?

A phase-field model and a GAN solve different problems.

## Phase-Field Model

A phase-field model describes the evolution of a physical system using continuous fields.

A simplified phase-field equation can be written as:

$$
\frac{\partial \phi}{\partial t}
=
-M
\frac{\delta F}{\delta \phi}
$$

where:

- $\phi$ is an order parameter
- $M$ is mobility
- $F$ is a free-energy functional

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

### Phase Field

$$
\text{Physics}
\rightarrow
\text{Evolution}
\rightarrow
\text{Microstructure}
$$

### GAN

$$
\text{Existing Microstructures}
\rightarrow
\text{Learning}
\rightarrow
\text{New Microstructures}
$$

A phase-field model is appropriate when the objective is to model the physical formation or evolution of a microstructure.

A GAN is useful when the objective is to rapidly generate new structures that resemble an existing population of microstructures.

These approaches are therefore complementary rather than direct replacements for each other.

---

# 6. GAN Does Not Replace Physics

The GAN does not automatically guarantee that a generated microstructure is physically valid.

The GAN learns the distribution of the training data:

$$
p_G(X)
\approx
p_{\mathrm{data}}(X)
$$

It does not inherently know physical properties such as:

$$
E,\qquad
k,\qquad
K,\qquad
\sigma_y
$$

or other material properties.

Therefore, generated structures should subsequently be evaluated using physics-based methods.

The complete workflow can be:

$$
\boxed{
\text{GAN}
\rightarrow
\text{Generated Microstructure}
\rightarrow
\text{Physics Simulation}
\rightarrow
\text{Effective Properties}
}
$$

This combines machine learning with computational physics.

---

# 7. GAN Architecture

The model consists of two neural networks:

1. Generator
2. Discriminator

The generator creates a microstructure:

$$
G(z,c)
=
X_{\mathrm{fake}}
$$

where:

- $z$ is a random latent vector
- $c$ is the condition
- $X_{\mathrm{fake}}$ is the generated microstructure

The discriminator attempts to distinguish real and generated microstructures.

---

# 8. Generator

The generator receives:

$$
z \in \mathbb{R}^{128}
$$

and a condition:

$$
c \in \mathbb{R}^{2}
$$

The combined input is:

$$
[z,c] \in \mathbb{R}^{130}
$$

The first fully connected layer maps this representation to:

$$
512 \times 4 \times 4 \times 2
$$

The tensor is then reshaped into:

$$
512 \times 4 \times 4 \times 2
$$

3D transposed convolutions progressively increase the spatial resolution:

$$
4 \times 4 \times 2
\rightarrow
8 \times 8 \times 4
$$

$$
8 \times 8 \times 4
\rightarrow
16 \times 16 \times 8
$$

$$
16 \times 16 \times 8
\rightarrow
32 \times 32 \times 16
$$

$$
32 \times 32 \times 16
\rightarrow
64 \times 64 \times 32
$$

$$
64 \times 64 \times 32
\rightarrow
128 \times 128 \times 64
$$

The final output is:

$$
X_{\mathrm{fake}}
\in
[-1,1]^{128 \times 128 \times 64}
$$

because the final layer uses the hyperbolic tangent activation:

$$
\tanh(x)
$$

---

# 9. Why 3D Convolution?

A microstructure is inherently three-dimensional.

Using 2D convolutions independently on individual slices would lose correlations between neighboring slices.

A 3D convolution operates on:

$$
x,y,z
$$

simultaneously.

A simplified 3D convolution can be represented as:

$$
Y(i,j,k)
=
\sum_{a,b,c}
W(a,b,c)
X(i+a,j+b,k+c)
$$

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

$$
X \in \mathbb{R}^{128 \times 128 \times 64}
$$

and progressively reduces its spatial resolution:

$$
128 \times 128 \times 64
\rightarrow
64 \times 64 \times 32
$$

$$
64 \times 64 \times 32
\rightarrow
32 \times 32 \times 16
$$

$$
32 \times 32 \times 16
\rightarrow
16 \times 16 \times 8
$$

$$
16 \times 16 \times 8
\rightarrow
8 \times 8 \times 4
$$

$$
8 \times 8 \times 4
\rightarrow
4 \times 4 \times 2
$$

The resulting features are flattened and passed through a linear layer.

The discriminator outputs one scalar:

$$
D(X) \in \mathbb{R}
$$

---

# 11. Adversarial Learning

The generator and discriminator are trained against each other.

The discriminator attempts to distinguish:

$$
X_{\mathrm{real}}
$$

from:

$$
X_{\mathrm{fake}}
$$

The generator attempts to produce structures that the discriminator considers realistic.

Conceptually:

$$
G
\leftrightarrow
D
$$

As training progresses:

- $D$ learns features that distinguish real and generated structures.
- $G$ learns features that make generated structures more similar to the training distribution.

---

# 12. Hinge Loss

This implementation uses hinge adversarial loss.

The discriminator loss is:

$$
L_D
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[
\max(0,1-D(x))
\right]
+
\mathbb{E}_{z\sim p(z)}
\left[
\max(0,1+D(G(z)))
\right]
$$

The generator loss is:

$$
L_G
=
-\mathbb{E}_{z\sim p(z)}
\left[
D(G(z))
\right]
$$

The discriminator does not use a sigmoid output because the hinge formulation works directly with the raw discriminator score.

---

# 13. Why Hinge Loss?

Hinge loss is commonly used in modern GAN architectures.

For real samples:

$$
L_{\mathrm{real}}
=
\max(0,1-D(x))
$$

For generated samples:

$$
L_{\mathrm{fake}}
=
\max(0,1+D(G(z)))
$$

The discriminator is encouraged toward:

$$
D(x)>1
$$

for real data and:

$$
D(G(z))<-1
$$

for generated data.

The generator attempts to increase:

$$
D(G(z))
$$

so that generated structures are considered more realistic.

---

# 14. Optimization

The model uses the Adam optimizer.

For a parameter $\theta$ and gradient:

$$
g_t = \nabla_\theta L_t
$$

Adam maintains first and second moment estimates:

$$
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

$$
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

followed by bias correction and parameter updates.

The implementation uses:

$$
\alpha = 2\times10^{-4}
$$

$$
\beta_1 = 0.5
$$

$$
\beta_2 = 0.999
$$

Adam is used because GAN training involves two simultaneously changing neural networks and can be difficult to optimize using ordinary gradient descent.

---

# 15. Data Processing

The original data contains binary values:

$$
X\in\{0,1\}
$$

The generator uses a final $\tanh$ activation, producing:

$$
X_{\mathrm{generated}}\in[-1,1]
$$

Therefore, real data is transformed using:

$$
X'=2X-1
$$

giving:

$$
0\rightarrow-1
$$

and:

$$
1\rightarrow+1
$$

This puts the real and generated data in the same numerical range.

---

# 16. Dataset Loading

The HDF5 files contain multiple microstructures.

Each HDF5 key corresponds to one 3D volume.

Instead of loading every microstructure into RAM, the dataset stores an index:

```text
(file path, HDF5 key)
