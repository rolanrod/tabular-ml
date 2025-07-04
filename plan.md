# Planning my Project

## Creating a Rust Library
A Rust library is a collection of code designed to be reused by other Rust programs. Unlike a binary crate (which produces an executable), a library crate produces a `.rlib` file that other projects can link to. Libraries are defined in a Rust project (called a crate) and organized using modules for structure

To create a Rust library,you need Rust installed (`rustup`). Then the following steps:

1) `cargo new <my_rust_lib>` --lib
    - The `--lib` flag tells Cargo to create a library crate
    - Creates a directory `<my_rust_lib>` with a `Cargo.toml` file and a `src` directory

2) `Cargo.toml` is a configuration file that specifies information about the library, including metadata (name, version, Rust edition) and dependencies. `src/lib.rs` is the main entry point of the library

3) The library can be compiled using `cargo build` and tested using `cargo test`, which will run the test in the `#[cfg(test)]` module


## Organizing Code with Modules
Rust uses modules to organize code within a crate. Modules help group related functionality and control visibility (public vs. private)

## Implementation timeline

1. Core data structures (matrices, datasets)
2. Basic algorithms (linear regression, classification)
3. Preprocessing utilities (normalization, encoding)
4. Model training/prediction interfaces
5. Evaluation metrics


Tree-based Models:
- Decision Trees (classification & regression)
- Random Forest
- Gradient Boosting

Clustering:
- K-Means
- Hierarchical Clustering
- DBSCAN

Dimensionality Reduction:
- PCA (Principal Component Analysis)
- LDA (Linear Discriminant Analysis)

Naive Bayes:
- Gaussian Naive Bayes
- Multinomial Naive Bayes

Ensemble Methods:
- Bagging
- AdaBoost
- Voting Classifiers

Model Selection & Validation:
- Cross-validation
- Grid search
- Train/validation/test splits