### 1. Finish Coding the Experimental Setup

**Objective:** Finalize the code required to run experiments testing different feature transforms and noise models for the graph diffusion process.

**Context:** The current model uses a GNN followed by a transformer for denoising. The central hypothesis is that the transformer's effectiveness comes from performing an operation similar to Kernel PCA in a feature space that is well-suited for graph data. To test this and related ideas, a flexible experimental framework is needed. This code should be capable of:

1. Applying a non-linear transform (like the `sigma_inverse` or logit function) to a graph's adjacency matrix to move it into a latent space.
2. Injecting noise into the graph representation, with the flexibility to add it either directly to the adjacency matrix or within the aforementioned latent space.
3. Deriving input features for the model (e.g., via eigendecomposition/PCA) from the noisy graph representation.
4. Training a model to denoise these features and reconstruct the original graph.

This setup will allow for systematically comparing different theoretical approaches to denoising.

---

### 2. Run Experiments with the `sigma_inverse` Transform

**Objective:** Conduct an experiment to test if defining the noise process within a latent, low-rank space (accessed via `sigma_inverse`) leads to effective denoising.

**Context:** The discussion proposed a theoretical model where a graph's adjacency matrix `S` is generated from a low-rank latent matrix `L` via a sigmoid-like function: `S = sigma(L)`. If this is true, the ideal space for denoising would be the low-rank `L` space itself, which can be recovered by `L = sigma_inverse(S)`.

The experiment to run is:

1. Start with a clean graph `S` and find its latent representation `L = sigma_inverse(S)`.
2. Define the forward diffusion process by adding noise directly to this latent matrix: `L_noisy(t) = L + noise(t)`. The noise is modeled as a low-rank perturbation `U U^T`.
3. Transform the noisy latent matrix back into a noisy graph: `S_noisy(t) = sigma(L_noisy(t))`.
4. Train the model to take `S_noisy(t)` as input, extract features, and predict the noise `U U^T` that was added to `L`.

Success in this experiment would provide strong evidence that the `sigma_inverse` space is a meaningful one for graph denoising and supports the Kernel PCA hypothesis.

---

### 3. Write Up the Mechanistic Explanation of Self-Attention for Denoising

**Objective:** Formalize and document the alternative hypothesis explaining how a self-attention layer could mechanistically learn to remove noise from graph embeddings.

**Context:** As an alternative to the high-level Kernel PCA theory, a more "bottom-up" explanation was proposed. This hypothesis suggests that the transformer directly learns an algorithm to isolate and remove noise components. If noise is modeled as adding perturbations `U U^T` in directions orthogonal to the main data manifold, the self-attention mechanism could learn to:

1. **Identify the Data Subspace:** Use its key and query projection matrices to learn the principal directions of the *clean* data. These directions could be encoded as biases or as the primary vectors in the learned matrices.
2. **Isolate Noise via Attention:** When an input embedding is processed, its alignment with these learned "data directions" will result in high attention scores. Components of the input that represent noise will be orthogonal and receive low attention.
3. **Reconstruct a Clean Output:** The value vectors, weighted by these attention scores, would effectively reconstruct the input by only using the parts that lie within the data subspace, thereby removing the noise.

This written explanation would serve as a concrete, testable hypothesis about the internal computations of the transformer.

---

### 4. Formulate a Noise Process Converging to an Erdos-Renyi Prior

**Objective:** Theoretically define a forward diffusion process in the latent space that is guaranteed to converge to a well-understood, uninformative graph distribution, such as an Erdos-Renyi (E-R) graph.

**Context:** A robust diffusion model should transform any input graph into a standard prior distribution (e.g., pure noise) from which sampling is easy. For graphs, an E-R graph, where every edge exists with a uniform probability `p`, is a good candidate for this prior.

The task is to connect the proposed latent space model to this prior. Given that `S = sigma(L)`, the elements of `S` can be interpreted as edge probabilities. The goal is to design a noise process for the latent matrix `L` (e.g., adding progressively larger Gaussian noise to its elements) and prove that the resulting edge probabilities in `S` converge to a constant `p`. This would provide a solid theoretical justification for the forward process and ensure that the final noised state is a proper, sampleable distribution.

---

### 5. Send the Leiden Paper on Community Detection

**Objective:** Share the "Leiden algorithm" paper with Shervin.

**Context:** During the discussion on how to analyze the effect of noise on graph structure, the Leiden and Louvain methods for community detection were mentioned. These algorithms often work by comparing the observed graph structure to a reference or null model (like a configuration model). The paper might provide useful concepts or tools for quantifying how noise degrades the community structure of a graph, which could be a valuable metric for evaluating the denoising process.

---

### 6. Use Captum for Interpretability (Shared Task)

**Objective:** Apply interpretability methods, such as "input gradient with noise" from the Captum library, to understand which features the transformer model uses to denoise the graph embeddings.

**Context:** To test the mechanistic hypothesis, it's crucial to know what the model is "looking at." This experiment aims to determine if the transformer focuses its attention on the top principal components of the embeddings (where the signal is presumed to be) or if it leverages information from the lower-variance components to distinguish signal from noise. The results would provide direct evidence for or against the proposed mechanism where the model learns to identify and project onto a clean data subspace.

---

### 7. Empirically Analyze DiGress Noise in the Latent Space (Shared Task)

**Objective:** Empirically measure the effect of a real-world graph noise process (DiGress) within the proposed `sigma_inverse` latent space.

**Context:** The current theory models noise as a low-rank addition `U U^T` in the latent space. This action item is to verify if this theoretical model matches reality. The experiment would involve:

1. Taking a clean graph `S_clean`.
2. Applying a single step of DiGress noise to get `S_noisy`.
3. Transforming both graphs into the latent space: `L_clean = sigma_inverse(S_clean)` and `L_noisy = sigma_inverse(S_noisy)`.
4. Calculating the difference: `Noise_empirical = L_noisy - L_clean`.
5. Analyzing the structure of `Noise_empirical`. Is it approximately low-rank? Does it resemble the `U U^T` model?

This will help bridge the gap between the theoretical noise model and the practical noise scheduling used in existing methods.

