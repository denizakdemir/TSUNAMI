Below is a reorganized and consolidated "ToDo" document for the TSUNAMI repository (denizakdemir/TSUNAMI), based on the comprehensive review provided. This document groups recommendations and tasks by category (Code Quality, Research Applicability, Algorithmic Rigor, Documentation, and Usability) and provides clear, actionable steps for improvement. Each section includes a brief rationale, specific actions, and, where applicable, code snippets or examples to guide implementation.

---

# TSUNAMI Repository Improvement ToDo List

## Code Quality

### 1. Refactor Large Functions for Better Modularity
   - **Rationale**: Some functions (e.g., transformer encoder, training loop) may be overly large, reducing clarity and maintainability.
   - **Action**: Identify and split large functions in `encoder.py` and `model.py` into smaller, reusable helper functions.
   - **Example**:
     ```python
     # Before (large function in encoder.py)
     def forward(self, x):
         # Long sequence of operations...

     # After
     def preprocess_input(self, x):
         return self.norm(x)

     def apply_attention(self, x):
         return self.attention(x, x, x)

     def forward(self, x):
         x = self.preprocess_input(x)
         x = self.apply_attention(x)
         return self.ffn(x)
     ```

### 2. Enforce Consistent Coding Style
   - **Rationale**: Ensure PEP8 compliance and maintain readability as the project grows.
   - **Action**: Integrate a linter (e.g., flake8) or formatter (e.g., black) into the development workflow. Update `.gitignore` and add a pre-commit hook or CI check.
   - **Example**: Add to `.pre-commit-config.yaml`:
     ```yaml
     - repo: https://github.com/psf/black
       rev: 23.3.0
       hooks:
         - id: black
     - repo: https://github.com/PyCQA/flake8
       rev: 6.0.0
       hooks:
         - id: flake8
     ```

### 3. Implement Continuous Integration (CI)
   - **Rationale**: Automate testing to catch regressions and ensure quality on each commit.
   - **Action**: Set up GitHub Actions to run tests (`pytest`) on pull requests and pushes. Include checks for style and documentation coverage.
   - **Example**: Add `.github/workflows/ci.yml`:
     ```yaml
     name: CI Pipeline
     on: [push, pull_request]
     jobs:
       test:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v3
           - name: Set up Python
             uses: actions/setup-python@v4
             with:
               python-version: '3.9'
           - name: Install dependencies
             run: pip install -r requirements.txt
           - name: Run tests
             run: pytest tests/
     ```

### 4. Improve Version Control Practices
   - **Rationale**: Enhance traceability and collaboration with better commit messages and branching.
   - **Action**: Encourage descriptive commit messages (e.g., "Add transformer encoder optimization") and use topic branches for new features. Document this in a `CONTRIBUTING.md`.

### 5. Enhance Model Persistence with Versioning
   - **Rationale**: Improve tracking of model experiments with versioning and metadata.
   - **Action**: Update the `save` method in `model.py` to include versioning and metadata.
   - **Example** (from earlier review):
     ```python
     def save(self, path: str, metadata: dict = None):
         import time
         version = int(time.time())
         full_metadata = {
             "version": version,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
             "architecture": {
                 "num_continuous": self.num_continuous,
                 "encoder_dim": self.encoder_dim,
                 "include_variational": self.include_variational
             },
             "user_metadata": metadata or {}
         }
         torch.save({
             "state_dict": self.state_dict(),
             "metadata": full_metadata
         }, f"{path}_v{version}.pt")
         with open(f"{path}_latest.txt", "w") as f:
             f.write(str(version))
     ```

## Research Applicability

### 1. Document Theoretical Foundations
   - **Rationale**: Provide clarity on the mathematical and scientific basis to enhance trust and usability for researchers.
   - **Action**: Add a whitepaper or detailed section in the README/wiki describing the mathematical formulation (e.g., loss functions, transformer architecture) and cite relevant literature (e.g., DeepHit paper).
   - **Example**: Include in README:
     ```
     ### Theoretical Foundations
     TSUNAMI extends DeepHit by using a tabular transformer encoder. The loss function combines a negative log-likelihood term for survival times with a ranking loss (alpha_rank > 0) and calibration term (alpha_calibration > 0). See [Lee et al., 2018] for the original DeepHit formulation.
     ```

### 2. Enhance Reproducibility with Reference Results
   - **Rationale**: Help users verify results and compare performance against benchmarks.
   - **Action**: Add reference results (e.g., training curves, metrics) in documentation or `vignettes/`. Include fixed random seeds in examples.
   - **Example**: Update an example script:
     ```python
     torch.manual_seed(42)
     np.random.seed(42)
     ```

### 3. Provide Real-World Datasets and Examples
   - **Rationale**: Expand applicability by showing how TSUNAMI works on public datasets.
   - **Action**: Include links or scripts to preprocess datasets like SEER or TCGA. Add a notebook in `vignettes/` demonstrating end-to-end analysis.
   - **Example**: Create `vignettes/real_data_example.ipynb` with steps for loading, preprocessing, and evaluating on a public dataset.

### 4. Improve Citation Practices
   - **Rationale**: Ensure credit is given and methods are grounded in prior work.
   - **Action**: Add in-line citations or references in code comments and documentation for techniques like SHAP, variational methods, etc.

## Algorithmic Rigor

### 1. Verify and Document Loss Derivations
   - **Rationale**: Ensure mathematical correctness and clarity of loss functions (e.g., ranking, calibration).
   - **Action**: Document the derivation of loss terms in comments or a separate math appendix. Test outputs against expected values on toy data.
   - **Example**: Add to `model.py`:
     ```python
     def compute_loss(self, outputs, targets):
         # Negative Log-Likelihood for survival times
         # L_NLL = -1/N * sum(log(P(T_i | X_i))) where T_i is censored/adjusted
         nll_loss = torch.nn.BCEWithLogitsLoss()(outputs['survival_probs'], targets['event_times'])
         # Ranking loss ensures higher risk for earlier events
         rank_loss = self.ranking_loss(outputs['risk_scores'], targets['event_times'])
         return nll_loss + self.alpha_rank * rank_loss
     ```

### 2. Evaluate Transformer Scalability
   - **Rationale**: Address potential quadratic complexity of self-attention for large datasets.
   - **Action**: Implement and test alternative attention mechanisms (e.g., Performer, Linformer) in `encoder.py`.
   - **Example** (Skeleton):
     ```python
     class LinearAttention(nn.Module):
         def forward(self, x):
             # Implement linear-complexity attention
             return self.attention(x, linear=True)
     ```

### 3. Expand Evaluation Metrics
   - **Rationale**: Provide more comprehensive metrics for survival analysis.
   - **Action**: Add metrics like concordance index, Brier score, and Restricted Mean Survival Time (RMST) to `metrics.py`.
   - **Example**:
     ```python
     def concordance_index(predictions, actual_times, events):
         # Implementation of Harrell's C-index
         pass
     ```

### 4. Validate Interpretability Methods
   - **Rationale**: Ensure feature importance and uncertainty methods are correct and useful.
   - **Action**: Test feature importance (SHAP, permutation) on synthetic data and document results. Add examples in `vignettes/`.

## Documentation

### 1. Expand README with Tutorial
   - **Rationale**: Lower barrier to entry with a detailed “Getting Started” guide.
   - **Action**: Add a step-by-step tutorial in the README or wiki, possibly linking to a Jupyter notebook.
   - **Example**: Include in README:
     ```
     ## Getting Started
     1. Install dependencies: `pip install -r requirements.txt`
     2. Run an example: `python examples/single_risk_example.py`
     3. See `vignettes/getting_started.ipynb` for a detailed walkthrough.
     ```

### 2. Generate HTML Documentation
   - **Rationale**: Provide an easily browsable API reference.
   - **Action**: Use Sphinx to generate HTML docs from docstrings. Host on GitHub Pages or ReadTheDocs.
   - **Example**: Add `docs/conf.py` and run `make html` in the docs folder.

### 3. Document Visualization and Interpretation
   - **Rationale**: Clarify how to use and interpret visualization tools.
   - **Action**: Add a section in README or wiki showing how to generate plots (e.g., survival curves, attention weights) and interpret them.
   - **Example**: Include in README:
     ```
     ## Visualization
     Use `model.visualize_survival_curves(X_test)` to plot survival functions. Example output can be found in `vignettes/visualization_example.ipynb`.
     ```

### 4. Create FAQs and Troubleshooting Guide
   - **Rationale**: Anticipate user issues and provide solutions.
   - **Action**: Add an FAQ section to README or a separate `TROUBLESHOOTING.md`, covering common errors (e.g., missing `cat_feat_info`).

## Usability

### 1. Publish Package on PyPI
   - **Rationale**: Simplify installation and distribution.
   - **Action**: Prepare and upload TSUNAMI to PyPI, ensuring `setup.py` is complete. Test installation with `pip install tsunami-survival`.
   - **Example**: Update `setup.py` with classifiers and long description.

### 2. Improve Error Handling and Messages
   - **Rationale**: Make errors more user-friendly.
   - **Action**: Catch common input errors (e.g., wrong data shapes) and provide clear messages. Use Python warnings for potential issues.
   - **Example**:
     ```python
     def fit(self, train_loader, ...):
         if not train_loader.dataset:
             raise ValueError("Train loader must contain a dataset. Check DataLoader initialization.")
         if self.num_continuous <= 0:
             warnings.warn("No continuous features detected; ensure data is correctly preprocessed.")
     ```

### 3. Add Command-Line Interface (CLI)
   - **Rationale**: Enable quick experimentation without coding.
   - **Action**: Implement a CLI using `argparse` or `click` to run training jobs from the command line.
   - **Example**:
     ```python
     import click

     @click.command()
     @click.option('--config', help='Path to config file')
     def train(config):
         # Load config and run training
         pass

     if __name__ == '__main__':
         train()
     ```

### 4. Create Contributor Guidelines
   - **Rationale**: Facilitate contributions from the community.
   - **Action**: Add `CONTRIBUTING.md` with coding style, testing instructions, and design principles.
   - **Example**: Include in `CONTRIBUTING.md`:
     ```
     ## How to Contribute
     1. Fork the repository.
     2. Create a topic branch for your feature.
     3. Run tests with `pytest tests/` and ensure style compliance with `black .`.
     4. Submit a pull request.
     ```

---

This ToDo list prioritizes immediate improvements (e.g., documentation, CI) and longer-term enhancements (e.g., new algorithms, PyPI publishing). It provides a roadmap for maintaining and expanding TSUNAMI while ensuring it remains a robust, user-friendly, and scientifically sound tool for survival analysis research.