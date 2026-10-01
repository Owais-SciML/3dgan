# 3D GAN for Microstructure Generation

## Overview

This project implements a 3D Generative Adversarial Network (GAN) for generating binary two-phase microstructures from 3D voxel data.

The input microstructures are represented as 3D binary arrays:

\[
X \in \{0,1\}^{128\times128\times64}
\]

where each voxel represents one of two phases.

The project uses a 3D convolutional GAN consisting of:

- A 3D Generator
- A 3D Discriminator
- 3D transposed convolutions
- 3D convolutions
- Batch normalization
- ReLU and LeakyReLU activations
- Adam optimization
- Hinge adversarial loss
- 90% training and 10% validation split
- Latent random noise
- One-hot conditioning

The generated output is converted into a binary 3D microstructure and saved as an HDF5 file and NumPy array.

---

# 1. Microstructure Representation

A microstructure describes the spatial arrangement of different material phases.

For a two-phase material, each voxel can be represented as

\[
X_{ijk}\in\{0,1\}
\]

where

- \(0\) represents phase 0
- \(1\) represents phase 1

The complete microstructure is therefore

\[
X\in\{0,1\}^{N_x\times N_y\times N_z}
\]

For this project:

\[
N_x=128,\qquad
N_y=128,\qquad
N_z=64
\]

Therefore each microstructure contains

\[
128\times128\times64=1,048,576
\]

voxels.

The microstructure contains spatial information, not just the total amount of each phase.

For example, two structures can have the same volume fraction but completely different:

- phase connectivity
- pore structure
- interface area
- characteristic length scale
- tortuosity
- anisotropy
- transport properties
- mechanical response

Therefore the spatial distribution of the voxels is important.

---

# 2. Physics of the Problem

A material microstructure affects its macroscopic physical properties.

For example, consider a two-phase material with thermal conductivity

\[
k_0
\]

for phase 0 and

\[
k_1
\]

for phase 1.

The effective thermal conductivity of the complete material is not generally equal to a simple volume-weighted average.

Instead,

\[
k_{\mathrm{eff}}
=
f(k_0,k_1,\text{microstructure})
\]

where the microstructure controls the paths through which heat can travel.

Similarly, the effective elastic properties can be written conceptually as

\[
E_{\mathrm{eff}}
=
f(E_0,E_1,\text{microstructure})
\]

and fluid transport properties can depend on

\[
K_{\mathrm{eff}}
=
f(\mu,\text{microstructure})
\]

where \(K_{\mathrm{eff}}\) is permeability and \(\mu\) is fluid viscosity.

Therefore the objective of microstructure generation is not simply to generate random binary images.

The generated structure should reproduce the important statistical and physical characteristics of real microstructures.

---

# 3. Why Generate Microstructures?

Computational materials research often requires a large number of microstructures.

A conventional workflow can be:

\[
\text{Microstructure}
\rightarrow
\text{Physical Simulation}
\rightarrow
\text{Material Properties}
\]

For example:

\[
\text{Microstructure}
\rightarrow
\text{Finite Element Analysis}
\rightarrow
E_{\mathrm{eff}}
\]

or

\[
\text{Microstructure}
\rightarrow
\text{Heat Transfer Simulation}
\rightarrow
k_{\mathrm{eff}}
\]

or

\[
\text{Microstructure}
\rightarrow
\text{Fluid Flow Simulation}
\rightarrow
K_{\mathrm{eff}}
\]

These simulations can be computationally expensive when thousands or millions of candidate microstructures are required.

A generative machine learning model provides another approach:

\[
z
\rightarrow
G(z)
\rightarrow
\text{Microstructure}
\]

Once trained, generating a new structure can be much cheaper than running a complete physical simulation or a detailed microstructure-generation algorithm.

---

# 4. Why Deep Learning?

Traditional computational models attempt to explicitly describe the physical mechanisms that create or evolve a microstructure.

Deep learning instead learns a mapping from examples.

Given a training dataset

\[
\mathcal{D}
=
\{X_1,X_2,\ldots,X_N\}
\]

the neural network attempts to learn the underlying distribution

\[
p_{\mathrm{data}}(X)
\]

without explicitly writing an analytical equation for the complete microstructure distribution.

The generator attempts to produce samples from an approximation

\[
p_G(X)
\approx
p_{\mathrm{data}}(X)
\]

This is useful when the microstructure is complicated and difficult to describe analytically.

---

# 5. Why GAN Instead of a Phase-Field Model?

A phase-field model and a GAN solve different problems.

## Phase-Field Model

A phase-field model describes the evolution of a physical system using continuous fields.

A simplified phase-field equation can have the form

\[
\frac{\partial \phi}{\partial t}
=
-M
\frac{\delta F}{\delta\phi}
\]

where

- \(\phi\) is an order parameter
- \(M\) is mobility
- \(F\) is a free-energy functional

The phase-field method can model physical processes such as:

- phase separation
- grain growth
- solidification
- precipitation
- coarsening
- interface evolution

The resulting microstructure is generated through a physical evolution model.

## GAN

A GAN does not explicitly solve the physical evolution equation.

Instead, it learns the statistical distribution of structures from existing examples.

Therefore:

Phase field:

\[
\text{Physics}
\rightarrow
\text{Evolution}
\rightarrow
\text{Microstructure}
\]

GAN:

\[
\text{Existing Microstructures}
\rightarrow
\text{Learning}
\rightarrow
\text{New Microstructures}
\]

A phase-field model is therefore more appropriate when the objective is to model the physical formation or evolution of a microstructure.

A GAN is useful when the objective is to rapidly generate new structures that resemble an existing population of microstructures.

---

# 6. GAN Does Not Replace Physics

The GAN in this project does not automatically guarantee that a generated microstructure is physically valid.

This distinction is important.

The GAN learns the distribution of the training data:

\[
p_G(X)\approx p_{\mathrm{data}}(X)
\]

It does not inherently know:

\[
E
\]

\[
k
\]

\[
K
\]

\[
\sigma_y
\]

or any other physical property.

Therefore, a generated structure should subsequently be evaluated using physics-based methods.

For example:

\[
\boxed{
\text{GAN}
\rightarrow
\text{Generated Microstructure}
\rightarrow
\text{Physics Simulation}
\rightarrow
\text{Effective Properties}
}
\]

This creates a useful combination of machine learning and computational physics.

---

# 7. Computational Physics Alternative

Another approach is to generate structures using computational models and then evaluate them.

Examples include:

- Monte Carlo methods
- Cellular automata
- Phase-field models
- Molecular dynamics
- Finite element methods
- Lattice-based methods
- Direct numerical simulation

These methods can provide physical interpretability but may require substantial computational resources depending on the problem.

Deep learning provides a data-driven alternative when a sufficiently large and representative dataset is available.

The choice depends on the objective.

| Method | Main purpose |
|---|---|
| Phase field | Simulate physical microstructure evolution |
| Monte Carlo | Statistical/thermodynamic sampling |
| FEM | Solve mechanical/thermal physical problems |
| CFD | Solve fluid-flow problems |
| Molecular dynamics | Atomistic-scale simulation |
| GAN | Learn and generate a distribution of structures |

These methods are complementary rather than direct substitutes.

---

# 8. GAN Architecture

The model consists of two neural networks.

## Generator

The generator creates a microstructure.

\[
G(z,c)=X_{\mathrm{fake}}
\]

where

- \(z\) is a random latent vector
- \(c\) is the condition
- \(X_{\mathrm{fake}}\) is the generated microstructure

The latent vector contains random information used to generate different structures.

In this implementation:

\[
z\in\mathbb{R}^{128}
\]

The condition is represented using a one-hot vector:

\[
c\in\mathbb{R}^{2}
\]

Therefore the generator input is

\[
[z,c]\in\mathbb{R}^{130}
\]

---

# 9. Generator Architecture

The generator starts with a fully connected layer.

\[
130
\rightarrow
512\times4\times4\times2
\]

The resulting tensor is reshaped into

\[
512\times4\times4\times2
\]

The spatial resolution is then progressively increased using 3D transposed convolutions.

\[
4\times4\times2
\rightarrow
8\times8\times4
\]

\[
8\times8\times4
\rightarrow
16\times16\times8
\]

\[
16\times16\times8
\rightarrow
32\times32\times16
\]

\[
32\times32\times16
\rightarrow
64\times64\times32
\]

\[
64\times64\times32
\rightarrow
128\times128\times64
\]

The final output is

\[
X_{\mathrm{fake}}
\in
[-1,1]^{128\times128\times64}
\]

because the final layer uses the hyperbolic tangent activation:

\[
\tanh(x)
\]

---

# 10. Why Transposed Convolution?

A normal convolution generally reduces or preserves spatial resolution depending on its parameters.

The generator needs to perform the opposite operation.

3D transposed convolution allows the network to progressively construct a high-resolution 3D structure from a low-dimensional latent representation.

Conceptually:

\[
\text{Latent Vector}
\rightarrow
\text{Low Resolution Features}
\rightarrow
\text{High Resolution Features}
\rightarrow
\text{3D Microstructure}
\]

The convolutional filters learn spatial patterns such as:

- local phase arrangements
- interfaces
- clusters
- connectivity
- larger-scale morphology

---

# 11. Discriminator

The discriminator receives either a real or generated microstructure.

Its task is to distinguish:

\[
X_{\mathrm{real}}
\]

from

\[
X_{\mathrm{fake}}
\]

The discriminator performs the reverse spatial transformation.

\[
128\times128\times64
\rightarrow
64\times64\times32
\]

\[
64\times64\times32
\rightarrow
32\times32\times16
\]

\[
32\times32\times16
\rightarrow
16\times16\times8
\]

\[
16\times16\times8
\rightarrow
8\times8\times4
\]

\[
8\times8\times4
\rightarrow
4\times4\times2
\]

The resulting features are flattened and passed to a linear layer.

The discriminator outputs one scalar:

\[
D(X)\in\mathbb{R}
\]

A larger value indicates that the discriminator considers the sample more realistic under the hinge-loss formulation.

---

# 12. Adversarial Learning

The generator and discriminator are trained against each other.

The discriminator attempts to distinguish real and generated structures.

The generator attempts to generate structures that the discriminator considers realistic.

This creates the adversarial process:

\[
G
\leftrightarrow
D
\]

The discriminator improves its representation of real microstructures while the generator improves its ability to reproduce the learned distribution.

---

# 13. Hinge Loss

This implementation uses hinge adversarial loss.

For the discriminator:

\[
L_D
=
E_{x\sim p_{\mathrm{data}}}
[
\max(0,1-D(x))
]
+
E_{z\sim p(z)}
[
\max(0,1+D(G(z)))
]
\]

The generator minimizes:

\[
L_G
=
-E_{z\sim p(z)}
[
D(G(z))
]
\]

There is no sigmoid activation at the discriminator output.

This is intentional because the hinge formulation operates directly on the discriminator's raw output.

---

# 14. Why Hinge Loss?

Binary cross-entropy is commonly used for GANs, but hinge loss is also widely used for adversarial training.

Hinge loss provides useful gradients when the discriminator is confidently classifying samples.

For the real sample:

\[
L_{\mathrm{real}}
=
\max(0,1-D(x))
\]

For the fake sample:

\[
L_{\mathrm{fake}}
=
\max(0,1+D(G(z)))
\]

The discriminator is encouraged to satisfy approximately:

\[
D(x)>1
\]

for real data and

\[
D(G(z))<-1
\]

for generated data.

The generator attempts to increase:

\[
D(G(z))
\]

so that generated structures appear realistic to the discriminator.

---

# 15. Optimization

The model uses the Adam optimizer.

The update is based on estimates of the first and second moments of the gradient.

For a parameter \(\theta\):

\[
g_t=\nabla_\theta L_t
\]

Adam maintains:

\[
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
\]

and

\[
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
\]

followed by bias correction and parameter updates.

The implementation uses:

\[
\alpha=2\times10^{-4}
\]

\[
\beta_1=0.5
\]

\[
\beta_2=0.999
\]

Adam is used because GAN training involves two simultaneously changing networks and can be difficult to optimize with ordinary gradient descent.

---

# 16. Data Processing

The original binary data is:

\[
X\in\{0,1\}
\]

The generator uses a final tanh activation, which produces values in:

\[
[-1,1]
\]

Therefore the real data is transformed using:

\[
X'
=
2X-1
\]

giving:

\[
0\rightarrow-1
\]

\[
1\rightarrow+1
\]

This puts the real training data into the same numerical range as the generator output.

---

# 17. Dataset Loading

The HDF5 files contain multiple microstructures.

Each HDF5 key corresponds to one 3D volume.

The code creates an index:

\[
(\text{file path},\text{HDF5 key})
\]

instead of loading every microstructure into RAM.

When a sample is requested, the corresponding HDF5 dataset is loaded.

This is important because one microstructure contains:

\[
1,048,576
\]

voxels.

Loading tens of thousands of full-resolution 3D structures simultaneously would require a very large amount of memory.

---

# 18. Train/Validation Split

The dataset is divided into:

\[
90\%
\]

training data and

\[
10\%
\]

validation data.

For \(N\) samples:

\[
N_{\mathrm{train}}=0.9N
\]

\[
N_{\mathrm{validation}}=0.1N
\]

The validation set is not used to update the network parameters.

It provides an independent set of examples for monitoring the behavior of the trained model.

---

# 19. Mini-Batch Training

The 90/10 split should not be confused with batch size.

For example, with 100 microstructures:

\[
90
\]

are used for training and

\[
10
\]

for validation.

The training data is then divided into smaller mini-batches.

For this project:

```text
BATCH_SIZE = 2
