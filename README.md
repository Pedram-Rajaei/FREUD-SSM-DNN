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
- a<b>SSM-DNN Hybrid Model</b>: Integrates the temporal dynamics of SSM with the classification capabilities of DNN.
- a<b>Latent Manifold Representation</b>: Learns a low-dimensional structure that represents the neural data across time.
- a<b>Multi-Scale Decoding</b>: Classifies task labels based on neural dynamics at different temporal scales, enabling robust trial-level predictions.
- a<b>Feature Importance with Integrated Gradients</b>: Assesses feature contributions within the manifold for biological interpretability.
