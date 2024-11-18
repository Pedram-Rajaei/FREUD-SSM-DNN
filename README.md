<h1>Inferring and Predicting Neural Dynamics via Generative-Discriminative Manifold Learning</h1>

<h2>Overview</h2>
<p>
This repository contains the code and models for the study <strong>"Inferring and Predicting Neural Dynamics via Generative-Discriminative Manifold Learning."</strong> 
This project presents a novel hybrid model that integrates <strong>state-space models (SSM)</strong> with <strong>deep neural networks (DNN)</strong> to enable both inference of latent neural dynamics and accurate classification 
of task-specific labels from high-dimensional time series data, such as neural recordings. By combining generative and discriminative approaches, the model uncovers latent manifold structures 
within neural data and predicts trial-level labels, advancing the fields of neural manifold learning and neural decoding.
</p>

<h3>This repository provides:</h3>

<h4>1. Model Definition and Methods for Learning and Inference</h4>
<p>
In neuroscience, the manifold hypothesis posits that complex, high-dimensional data, such as neural recordings, lie on a lower-dimensional, non-linear manifold. Identifying this manifold and 
decoding latent neural states is crucial for understanding the underlying dynamics associated with motor control or cognitive processes.
</p>
<p>
This research introduces a model that combines <strong>state-space models (SSM)</strong> and <strong>deep neural networks (DNN)</strong> to characterize this manifold. The state-space model governs temporal dynamics, 
capturing the evolution of latent states, while the deep neural network leverages these states for trial-level label prediction. By integrating these components, the model performs 
manifold inference and label prediction simultaneously, providing an interpretable and scalable approach to neural decoding.
</p>

<h4>2. Toolset Functions and Structure</h4>
<p>
The repository includes a comprehensive suite of functions for preprocessing neural data, implementing the particle filter for state inference, training the DNN for classification, 
and running the Expectation-Maximization (EM) algorithm. These toolsets allow researchers to:
</p>
<ul>
  <li>Process high-dimensional neural time-series data efficiently.</li>
  <li>Use particle filtering for inference of latent states, which are otherwise unobservable.</li>
  <li>Optimize model parameters iteratively through the EM algorithm, ensuring that both the generative and discriminative components adapt to the data.</li>
</ul>
<p>
The modular codebase is structured to support flexibility and extendability for other machine learning tasks involving manifold learning and state inference.
</p>

<h4>3. Simulation Analysis</h4>
<p>
To validate the approach, the repository provides simulation frameworks for synthetic data generation and analysis. These simulations are critical for:
</p>
<ul>
  <li>Testing the performance of the SSM-DNN model under controlled conditions.</li>
  <li>Evaluating the accuracy of state inference and label prediction.</li>
  <li>Exploring the role of key hyperparameters (e.g., number of particles, dimensionality of the latent space) in the model's performance.</li>
</ul>
<p>
The simulation scripts demonstrate the robustness of the hybrid model in learning complex dynamics and adapting to varying levels of noise and variability.
</p>

<h4>4. Application in Neural Data</h4>
<p>
The model is applied to real neural data, such as electrocorticography (ECoG) or electroencephalography (EEG), to decode neural dynamics associated with tasks like hand movement and resting states. 
By inferring a latent manifold from these recordings, the model provides insights into the temporal evolution of neural activity and its relationship to task-specific labels.
</p>
<p>
For example, in the <strong>AJILE12 dataset</strong>, the model achieves state-of-the-art classification accuracy, outperforming benchmarks in neural decoding. Additionally, the use of 
Integrated Gradients enables feature importance analysis, highlighting significant neural components that contribute to task predictions.
</p>

<h2>Key Contributions</h2>

<ul>
  <li>
    <strong>SSM-DNN Hybrid Model:</strong> Integrates the temporal dynamics of SSM with the predictive capabilities of DNN. 
    The framework is adaptable for both classification and regression tasks, making it suitable for a wide range of neural data analysis problems.
  </li>
  <li>
    <strong>Latent Manifold Representation:</strong> Learns a low-dimensional structure that effectively represents neural data across time, 
    capturing underlying dynamics and providing insights into task-specific or condition-specific neural activity.
  </li>
  <li>
    <strong>Temporal Scale Integration:</strong> Combines information across different time scales, enabling the model to leverage both short-term 
    and long-term dynamics for improved task-specific prediction.
  </li>
  <li>
    <strong>Feature Sorting via Integrated Gradient Importance Ranking:</strong> Utilizes Integrated Gradients to rank features based on their importance, 
    allowing the identification of critical neural components within the manifold for interpretability and biological relevance.
  </li>
</ul>

<h2>Model Definition and Methods for Learning and Inference</h2>
<p>
This framework is built on two fundamental components:
</p>
<ul>
  <li>
    <strong>State-Space Model (SSM):</strong> A generative model that governs the temporal evolution of the latent states <em>X<sub>k</sub></em>, capturing dynamics in a reduced-dimensional space. 
    The SSM effectively models the temporal dependencies, providing a rich understanding of how neural states evolve over time.
  </li>
  <li>
    <strong>Deep Neural Network (DNN):</strong> A predictive model that leverages the latent states for both classification and regression tasks, optimizing the manifold for accurate 
    and interpretable predictions. The DNN captures task-specific features, mapping them to desired outputs such as task labels or regression targets.
  </li>
</ul>
<p>
By integrating these components, the model simultaneously performs <strong>manifold learning</strong> to uncover latent structures and <strong>prediction tasks</strong> to infer state dynamics 
or generate outputs effectively.
</p>

<div style="text-align: center; margin-top: 20px;">
  <!-- Image Section -->
  <img src="https://github.com/Pedram-Rajaei/FREUD-SSM-DNN/blob/main/Images/A_clear_and_hierarchical_2D_graphical_representati.png?raw=true" alt="Hierarchical 2D Graphical Representation" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;">
  
  <!-- Legend Section -->
  <div style="text-align: center; margin-top: 15px; font-family: Arial, sans-serif; font-size: 14px;">
    <strong>Legend:</strong>
    <ul style="list-style-type: none; padding: 0; text-align: left; display: inline-block;">
      <li><span style="color: green; font-weight: bold;">● Green:</span> Neural Data (<em>Y</em>) - Represents the input time-series data, such as neural recordings or EEG signals.</li>
      <li><span style="color: orange; font-weight: bold;">● Orange:</span> Latent States (<em>X</em>) - Represents the lower-dimensional manifold capturing the temporal dynamics inferred by the model.</li>
      <li><span style="color: blue; font-weight: bold;">● Blue:</span> Task Predictions (<em>L</em>) - Represents the output of the model, such as classification or regression results.</li>
    </ul>
  </div>
</div>



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
