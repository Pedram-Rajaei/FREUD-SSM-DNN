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
  <img src="https://github.com/Pedram-Rajaei/FREUD-SSM-DNN/blob/main/Images/A_clear_and_hierarchical_2D_graphical_representati.png?raw=true" alt="Hierarchical 2D Graphical Representation" style="max-width: 600px; height: 450px; border: 1px solid #ddd; border-radius: 8px;">
  
  <!-- Legend Section -->
  <div style="text-align: center; margin-top: 15px; font-family: Arial, sans-serif; font-size: 14px;">
    <strong>Legend:</strong>
    <ul style="list-style-type: none; padding: 0; text-align: left; display: inline-block;">
      <li><span style="color: green; font-weight: bold;">Green:</span> Neural Data (<em>Y</em>) - Represents the input time-series data, such as neural recordings or EEG signals.</li>
      <li><span style="color: orange; font-weight: bold;">Orange:</span> Latent States (<em>X</em>) - Represents the lower-dimensional manifold capturing the temporal dynamics inferred by the model.</li>
      <li><span style="color: blue; font-weight: bold;">Blue:</span> Task Predictions (<em>L</em>) - Represents the output of the model, such as classification or regression results.</li>
    </ul>
  </div>
</div>


<h2>Mathematical Formulation</h2>
<p>
This model is designed to analyze <strong>trial-level neural data</strong> to uncover latent dynamics and predict task-specific labels. 
Each trial corresponds to a time-segmented sequence of neural activity recorded during a specific experimental condition, such as a task or stimulus presentation. 
The model employs a combination of <strong>state-space modeling (SSM)</strong> and a <strong>deep neural network (DNN)</strong> to achieve this.
</p>

<h3>Trial Data Integration</h3>
<ul>
  <li>Neural recordings (<em>Y<sub>k</sub></em>) across multiple trials are used to train the model.</li>
  <li>Each trial is defined as a sequence of observations (<em>Y<sub>k</sub></em>) recorded over discrete time steps (<em>k</em>) and associated with a specific task or experimental condition.</li>
  <li>The model processes each trial’s latent dynamics (<em>X<sub>k</sub></em>) to predict its label (<em>l</em>), which represents the task or condition under which the data was recorded.</li>
</ul>

<h3>Transition Equation (State-Space Model - SSM)</h3>
<p>
The temporal evolution of the latent states is governed by the <strong>transition equation</strong>:
</p>
<div style="text-align: center;">
  <em style="text-align: center">X<sub>k+1</sub> | X<sub>k</sub> ∼ f<sub>ψ</sub>(X<sub>k</sub>, ε<sub>k</sub>), ε<sub>k</sub> ∼ N(0, R)</em>
</div>
<p>
Here:
</p>
<ul>
  <li><em>f<sub>ψ</sub></em>: A function capturing the temporal dependencies in the latent states, parameterized by <em>ψ</em>.</li>
  <li><em>ε<sub>k</sub></em>: Process noise with covariance <em>R</em>, accounting for uncertainty in the state evolution.</li>
</ul>
<p>
This equation ensures that the latent states are temporally coherent and evolve in a way that reflects the dynamics of the underlying neural processes.
</p>

<h3>Observation Equation</h3>
<p>
The relationship between the latent states and the recorded neural data (<em>Y<sub>k</sub></em>) is defined by the <strong>observation equation</strong>:
</p>
<div style="text-align: center; margin: 20px 0;">
  <em>Y<sub>k</sub> | X<sub>k</sub> ∼ g<sub>ϕ</sub>(X<sub>k</sub>, v<sub>k</sub>), v<sub>k</sub> ∼ N(0, Q)</em>
</div>
<p>
Here:
</p>
<ul>
  <li><em>g<sub>ϕ</sub></em>: A mapping function that relates latent states to observed data, parameterized by <em>ϕ</em>.</li>
  <li><em>v<sub>k</sub></em>: Observational noise with covariance <em>Q</em>, accounting for variability in the recorded data.</li>
</ul>
<p>
This equation allows the model to map latent states (<em>X<sub>k</sub></em>) to neural recordings (<em>Y<sub>k</sub></em>), capturing how observed neural signals are generated.
</p>

<h3>Classification Equation</h3>
<p>
The sequence of latent states across an entire trial (<em>X<sub>0</sub>, X<sub>1</sub>, ..., X<sub>K</sub></em>) is processed by a <strong>deep neural network (DNN)</strong> to predict the trial label (<em>l</em>):
</p>
<div style="text-align: center; margin: 20px 0;">
  <em>l | X<sub>0</sub>, ..., X<sub>K</sub> ∼ h<sub>ϕ</sub>(X<sub>0:K</sub>)</em>
</div>
<p>
Here:
</p>
<ul>
  <li><em>h<sub>ϕ</sub></em>: The DNN parameterized by <em>ϕ</em>, which learns to discriminate between task-specific labels based on the trajectory of latent states over time.</li>
  <li><em>l</em>: The predicted label, corresponding to the task or experimental condition.</li>
</ul>
<p>
The DNN complements the SSM by leveraging temporal dependencies in the latent states to optimize the representation for accurate label prediction.
</p>

<h3>Biological Relevance</h3>
<p>
This framework is particularly suited for neuroscience research, where latent neural states are often inferred to understand cognitive processes or motor control. For example:
</p>
<ul>
  <li><strong>Motor Control:</strong> Latent states (<em>X<sub>k</sub></em>) may correspond to brain activity patterns associated with hand movement.</li>
  <li><strong>Task-Specific Decoding:</strong> Labels (<em>l</em>) can represent tasks (e.g., left-hand vs. right-hand movement), and the model predicts these based on neural data.</li>
</ul>
<p>
By combining generative (SSM) and discriminative (DNN) approaches, the model simultaneously:
</p>
<ul>
  <li>Infers latent manifold structures.</li>
  <li>Decodes task-relevant information from neural recordings.</li>
</ul>



<h2>Code Structure and Documentation</h2>
The codebase is organized into the following modules:

- ```data_processing.py:``` Handles data loading and preprocessing, including reading CSV files and preparing time-series neural data.
- ```cnn1d_model.py:``` Defines a 1D CNN architecture that processes latent neural states for binary classification tasks.
- ```particle_filter.py:``` Implements the particle filtering algorithm to estimate the latent states over time.
- ```em_algorithm.py:``` Contains the Expectation-Maximization (EM) algorithm, which iteratively refines model parameters using E-step (particle filtering), M-step (parameter updates), and NN-step (DNN training).
- ```utils.py:``` Utility functions for data handling, training, and visualizing neural dynamics.

<h2>Methodology</h2>
<h2>EM Algorithm and Training Procedure</h2>
<p>
The <strong>Expectation-Maximization (EM)</strong> algorithm is a core component of this research, enabling the iterative optimization of model parameters under the SSM-DNN framework. 
This approach ensures that both the <strong>latent state dynamics</strong> and the <strong>task-specific predictions</strong> are accurately captured by maximizing the <strong>full likelihood</strong> of the observed data.
</p>

<h3>Objective: Full Likelihood</h3>
<p>
The full likelihood combines contributions from:
</p>
<ol>
  <li><strong>State Transitions:</strong> The probability of transitioning between latent states <em>X<sub>k</sub></em>, governed by the state-space model (SSM):
    <div style="text-align: center; margin: 10px 0;">
      <em>P(X<sub>k</sub> | X<sub>k-1</sub>; A, B, R)</em>
    </div>
    Here, <em>A</em>, <em>B</em> are the state transition matrices, and <em>R</em> is the process noise covariance.
  </li>
  <li><strong>Observations:</strong> The probability of the observed neural data <em>Y<sub>k</sub></em> given the latent states:
    <div style="text-align: center; margin: 10px 0;">
      <em>P(Y<sub>k</sub> | X<sub>k</sub>; C, D, Q)</em>
    </div>
    Here, <em>C</em>, <em>D</em> are the observation matrices, and <em>Q</em> is the observation noise covariance.
  </li>
  <li><strong>Task Predictions:</strong> The probability of predicting the trial label <em>l</em> given the latent state sequence <em>X<sub>0</sub>, ..., X<sub>K</sub></em> via the DNN:
    <div style="text-align: center; margin: 10px 0;">
      <em>P(l | X<sub>0:K</sub>; φ)</em>
    </div>
    Here, <em>φ</em> represents the parameters of the DNN.
  </li>
</ol>
<p>
The objective of the EM algorithm is to maximize the combined likelihood:
</p>
<div style="text-align: center; margin: 20px 0;">
  <em>ℒ = ∑<sub>trials</sub> [ log P(X<sub>0</sub>) + ∑<sub>k=1</sub><sup>K</sup> log P(X<sub>k</sub> | X<sub>k-1</sub>) + ∑<sub>k=1</sub><sup>K</sup> log P(Y<sub>k</sub> | X<sub>k</sub>) + log P(l | X<sub>0:K</sub>) ]</em>
</div>

<h3>Training Steps</h3>
<ol>
  <li><strong>Initialization:</strong>
    <ul>
      <li>Initialize latent states (<em>X<sub>0</sub></em>), noise covariances (<em>Q, R</em>), and transition matrices (<em>A, B, C, D</em>).</li>
      <li>Set initial parameters of the DNN classifier (<em>φ</em>).</li>
    </ul>
  </li>
  <li><strong>E-Step (Particle Filtering):</strong>
    <ul>
      <li>Use a <strong>particle filter</strong> to infer latent states (<em>X<sub>k</sub></em>) for each trial.</li>
      <li>Particles are propagated based on the SSM dynamics and weighted using the observed data (<em>Y<sub>k</sub></em>).</li>
      <li>This step generates state estimates that represent the latent manifold structure.</li>
    </ul>
  </li>
  <li><strong>M-Step (Parameter Update):</strong>
    <ul>
      <li>Update the <strong>transition matrices</strong> (<em>A, B</em>) and <strong>observation matrices</strong> (<em>C, D</em>) using maximum likelihood estimation.</li>
      <li>Recalculate covariances (<em>Q, R</em>) to reflect updated process and observation noise.</li>
    </ul>
  </li>
  <li><strong>NN-Step (DNN Training):</strong>
    <ul>
      <li>Train the DNN on the particle-filtered states to refine the latent representation and improve trial label predictions.</li>
      <li>The DNN adjusts its parameters (<em>φ</em>) to optimize classification or regression tasks.</li>
    </ul>
  </li>
  <li><strong>Convergence:</strong>
    <ul>
      <li>Repeat the steps until the likelihood stabilizes, indicating parameter convergence.</li>
    </ul>
  </li>
</ol>

<h3>Pseudocode for the EM Algorithm</h3>
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px;">
# Initialize parameters
Initialize X_0, A, B, C, D, Q, R, φ (DNN parameters)
Set convergence_threshold = ε
Set max_iterations = N
likelihood_previous = -inf

# Begin EM iterations
for iteration in range(max_iterations):
    # E-Step: Particle Filtering
    for trial in trials:
        for time_step in range(steps):
            Propagate particles using SSM dynamics
            Update particle weights using observed data Y_k
        Infer latent states X_k using resampled particles

    # M-Step: Update Parameters
    Update transition matrices (A, B) using maximum likelihood
    Update observation matrices (C, D) using inferred states
    Recalculate process (R) and observation (Q) covariances

    # NN-Step: Train DNN
    Train DNN using particle-filtered latent states X_0:K
    Update φ to improve label prediction

    # Evaluate likelihood
    likelihood_current = Calculate full likelihood (ℒ)
    if abs(likelihood_current - likelihood_previous) < convergence_threshold:
        break
    likelihood_previous = likelihood_current

# Return optimized parameters
Return A, B, C, D, Q, R, φ
</pre>
<p>
The EM algorithm combines generative and discriminative modeling to infer latent structures, optimize parameters, and predict task-specific labels, making it a robust tool for neural data analysis.
</p>

<h2>Feature Sorting Through Integrated Gradient Ranking</h2>
<p>
The <strong>Integrated Gradients (IG)</strong> method provides a principled approach to quantify feature importance in neural networks. 
By evaluating how much each input feature contributes to the model’s output, this method sheds light on the underlying neural components that drive task prediction. 
This is particularly valuable in understanding and interpreting neural data in a biologically meaningful way.
</p>

<h3>Overview of Integrated Gradients</h3>
<p>
Integrated Gradients measure feature importance by computing the integral of gradients of the model’s output with respect to its input, along a path from a baseline input (typically zero) to the actual input. 
It ensures that the feature attributions are consistent and satisfy the axioms of sensitivity and implementation invariance.
</p>

<h3>Mathematical Representation</h3>
<p>
For a model <em>F</em> and an input <em>x</em>, the Integrated Gradients for the <em>i</em>-th feature are defined as:
</p>
<div style="text-align: center; margin: 20px 0;">
  <em>IG<sub>i</sub>(x) = (x<sub>i</sub> - x<sub>i</sub><sup>baseline</sup>) 
  ∫<sub>α=0</sub><sup>1</sup> 
  (∂F(x<sup>baseline</sup> + α · (x - x<sup>baseline</sup>)) / ∂x<sub>i</sub>) dα</em>
</div>
<p>
Where:
</p>
<ul>
  <li><em>x<sup>baseline</sup></em>: The baseline input, often chosen as a vector of zeros or the mean of the input data.</li>
  <li><em>x</em>: The actual input.</li>
  <li><em>α</em>: A scaling factor that interpolates between the baseline and the actual input.</li>
  <li><em>∂F / ∂x<sub>i</sub></em>: The gradient of the model’s output with respect to the <em>i</em>-th feature.</li>
</ul>

<h3>Interpretation</h3>
<ul>
  <li>
    <strong>Feature Ranking:</strong> The magnitude of <em>IG<sub>i</sub>(x)</em> indicates the importance of the <em>i</em>-th feature. Features with larger magnitudes contribute more significantly to the output.
  </li>
  <li>
    <strong>Cumulative Contribution:</strong> The sum of Integrated Gradients across all features approximates the difference between the model’s output for the actual input and the baseline:
    <div style="text-align: center; margin: 20px 0;">
      <em>∑<sub>i</sub> IG<sub>i</sub>(x) ≈ F(x) - F(x<sup>baseline</sup>)</em>
    </div>
  </li>
  <li>
    <strong>Visualization:</strong> Integrated Gradients can be visualized as heatmaps or importance rankings to highlight which neural components (e.g., neurons, channels, or time steps) are most influential for a specific task.
  </li>
</ul>

<h3>Biological Relevance</h3>
<p>
When applied to neural data, Integrated Gradients enable:
</p>
<ul>
  <li><strong>Identification of Critical Neural Features:</strong> Pinpointing specific neurons, time points, or frequency bands that are most relevant for decoding task labels.</li>
  <li><strong>Interpretability in Neural Dynamics:</strong> Understanding how latent states or neural components interact to influence the model's predictions.</li>
</ul>

<h3>Advantages of Integrated Gradients</h3>
<ul>
  <li><strong>Sensitivity:</strong> Accurately attributes importance to features that affect the model's output.</li>
  <li><strong>Consistency:</strong> Provides the same attributions for models with equivalent functions, ensuring robustness across different architectures.</li>
  <li><strong>Biological Insight:</strong> Facilitates the interpretation of complex neural dynamics in a task-relevant context.</li>
</ul>

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
