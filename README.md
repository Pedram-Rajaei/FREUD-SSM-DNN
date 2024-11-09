<h1>Inferring and Predicting Neural Dynamics via Generative-Discriminative Manifold Learning</h1>

<h2>Overview</h2>
This repository contains the code and models for the study <b>"Inferring and Predicting Neural Dynamics via Generative-Discriminative Manifold Learning."</b> This project presents a novel hybrid model that integrates <b>state-space models (SSM)</b> with <b>deep neural networks (DNN)</b> to enable both inference of latent neural dynamics and accurate classification of task-specific labels from high-dimensional time series data, specifically neural recordings. By combining generative and discriminative approaches, the model uncovers latent manifold structures within neural data and predicts trial-level labels, advancing the fields of neural manifold learning and neural decoding.

This repository provides:

- A <b>1D Convolutional Neural Network (CNN)</b> architecture for classification on latent neural dynamics.
- A <b>particle filter</b> for inference of time-evolving latent states.
- An <b>Expectation-Maximization (EM)</b> algorithm to iteratively estimate and update model parameters.

<h2>Research Context</h2>
In neuroscience, the manifold hypothesis posits that complex, high-dimensional data, such as neural recordings, lie on a lower-dimensional, non-linear manifold. Identifying this manifold and decoding the latent neural states is crucial for understanding underlying neural dynamics, such as those associated with motor control or cognitive processes. This research introduces a model that combines <b>SSM</b> and <b>DNN</b> to characterize this manifold, linking neural dynamics to specific tasks or experimental conditions (e.g., hand movement or resting states) by jointly performing manifold inference and label prediction.

<h2>Key Contributions</h2>

- <b>SSM-DNN Hybrid Model</b>: Integrates the temporal dynamics of SSM with the classification capabilities of DNN.
- <b>Latent Manifold Representation</b>: Learns a low-dimensional structure that represents the neural data across time.
- <b>Multi-Scale Decoding</b>: Classifies task labels based on neural dynamics at different temporal scales, enabling robust trial-level predictions.
- <b>Feature Importance with Integrated Gradients</b>: Assesses feature contributions within the manifold for biological interpretability.

<h2>Model Architecture</h2>
This framework is built on two fundamental components:

- <b>State-Space Model (SSM):</b> A generative model that governs the temporal evolution of the latent states X<sub>k</sub>, capturing dynamics in a reduced-dimensional space.
- <b>Deep Neural Network (DNN):</b> A discriminative model that leverages the latent states to classify task-specific labels, optimizing the manifold for prediction accuracy.

<h2>Mathematical Formulation</h2>
The model operates on trial-level neural data and label sequences. Given a sequence of latent states X<sub>k</sub> and observations Y<sub>k</sub>, the evolution and observation equations are as follows:

 - <b>Transition Equation: (SSM):</b> <div align="center"><b><i>X<sub>k+1</sub> | X<sub>k</sub> ∼ f<sub>ψ</sub>(X<sub>k</sub>,     ϵ<sub>k</sub>), ϵ<sub>k</sub> ∼ N(0, R)</i></b></div>
<br>Here, <b><i>f<sub>ψ</sub></i></b> captures the temporal evolution of the latent states with process noise <b><i>ϵ<sub>k</sub></i></b> governed by covariance <b><i>R</i></b>.
 - <b>Observation Equation:</b> <div align="center"><b><i>Y<sub>k</sub> | X<sub>k</sub> ∼ g<sub>ϕ</sub>(X<sub>k</sub>, v<sub>k</sub>),     v<sub>k</sub> ∼ N(0, Q)</i></b></div>
<br>The mapping <b><i>g<sub>ϕ</sub></i></b> relates the latent states to neural observations <b><i>Y<sub>k</sub></i></b>, with observational noise and covariance <b><i>Q</i></b>.
 - <b>Classification Equation:</b> <div align="center"><b><i>l | X<sub>0</sub>, ..., X<sub>K</sub> ∼ h<sub>ϕ</sub>(X<sub>0:K</sub>)</i></b></div>
<br>The DNN, represented by <b><i>h<sub>ϕ</sub></i></b>, processes the latent state trajectory <b><i>X<sub>0:K</sub></i></b> to predict the label <b><i>l</i></b> associated with the task or condition.

<br>The SSM component captures temporal dependencies, while the DNN component leverages these dynamics for label prediction, optimizing the latent representation for both inference and discrimination.

<h2>Code Structure and Documentation</h2>
The codebase is organized into the following modules:

- ```data_processing.py:``` Handles data loading and preprocessing, including reading CSV files and preparing time-series neural data.
- ```cnn1d_model.py:``` Defines a 1D CNN architecture that processes latent neural states for binary classification tasks.
- ```particle_filter.py:``` Implements the particle filtering algorithm to estimate the latent states over time.
- ```em_algorithm.py:``` Contains the Expectation-Maximization (EM) algorithm, which iteratively refines model parameters using E-step (particle filtering), M-step (parameter updates), and NN-step (DNN training).
- ```utils.py:``` Utility functions for data handling, training, and visualizing neural dynamics.

<h2>Methodology</h2>
<h3>EM Algorithm and Training Procedure</h3>
The EM algorithm is essential to this research, iteratively estimating model parameters to maximize data likelihood under the SSM-DNN framework. The algorithm includes the following steps:

<h4>1. Initialization:<h4></h4>
The latent state, noise covariances, and transition matrices are initialized.

<h4>2. E-Step (Particle Filtering):</h4>
Latent states X<sub>k</sub> are inferred using a particle filter for each trial, leveraging the SSM dynamics. The particle filter iteratively updates particles based on neural data, generating state estimates that represent the underlying manifold structure.

<h4>3. M-Step (Parameter Update):</h4>
Model parameters are updated using maximum likelihood estimation. Transition (A, B) and observation (C, D) matrices are optimized based on inferred states, while covariances <I>Q</I> and <I>R</I> are recalculated to account for process and observation noise.

<h4>4. NN-Step (DNN Training):</h4>
The DNN classifier is trained using updated particle-filtered states, refining the latent representation to improve label prediction.

<h4>5. Convergence:</h4>
This process repeats until the model achieves stable likelihood values, indicating convergence.

<h3>Feature Importance via Integrated Gradients</h3>
The Integrated Gradients method evaluates feature importance by assessing the contribution of each input feature to the model’s output, providing insights into which neural components are critical for task prediction.

<h2>Dataset and Experimentation</h2>
The model was validated using the <b>AJILE12 dataset</b> of ECoG recordings from participants performing motor and rest tasks. A thorough evaluation was conducted using 10-fold cross-validation, yielding an F1 score of <b>0.75 ± 0.01</b>, outperforming benchmark methods such as HTNet by 5%.

The optimal latent dimensionality for EEG data was identified as 15, maximizing the decoding performance and interpretability of the model’s manifold representation.

<h2>Installation</h2>
<h3>Prerequisites</h3>
The project requires the following libraries:

- ```torch```
- ```numpy```
- ```pandas```
- ```scipy```
- ```matplotlib```
- ```seaborn```
<br>
Install them via:
<br>
<pre>pip install torch numpy pandas scipy matplotlib seaborn</pre>

<h2>Running the Model</h2>
<b>1. Data Preparation:</b> Store neural time-series data in the following directory structure on Google Drive:
<pre>
Google Drive
└── My Drive
    └── Behavior
        ├── ctl   # Control condition data files
        └── mdd   # Task-specific data files
</pre>
<b>2. Model Training and EM Iteration:</b> Run the primary training script:
<pre>
python main.py
</pre>
This script loads data, trains the CNN classifier, and iteratively updates model parameters using EM steps.

<h2>Example Usage</h2>
<pre><code>
# Import necessary modules
from cnn1d_model import CNN1D
from em_algorithm import EMAlgorithm

#Configure model parameters
dim_state, dim_obs = 1, 1
R, R0 = np.array([1]), np.array([1])
n_steps, num_particles, num_trials, max_iter = 360, 500, 23, 10

#Initialize and execute the EM Algorithm
em_algo = EMAlgorithm(Y, label, dim_state, dim_obs, simulation_mode=1, update_params_mode=[1, 2, 2, 1],
                      mode=1, R=R, R0=R0, n_steps=n_steps, num_particles=num_particles,
                      num_trials=num_trials, max_iter=max_iter)
em_algo.run_nn()  # Neural network initialization
em_algo.run_em()  # Run EM iteration
</code></pre>

<h2>Results and Interpretability</h2>

The model significantly improved decoding accuracy and interpretability:

- <b>Classification Performance:</b> The SSM-DNN model achieved an F1 score of 0.75 ± 0.01, indicating robust decoding accuracy.
- <b>Feature Relevance:</b> Using Integrated Gradients, the model highlighted the role of the pre-central gyrus in task execution, aligning with known motor control areas in the cortex.

These findings underscore the model’s utility in both decoding neural states and elucidating neural mechanisms underlying specific tasks.

<h2>Citation</h2>

If this work contributes to your research, please cite as follows:

<h2>Acknowledgments</h2>

We thank the neuroscience and machine learning communities for foundational methods and datasets that contributed to this work. Special thanks to the contributors of the AJILE12 dataset and to those who developed the Integrated Gradients methodology.
