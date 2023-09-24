//! Sparse Linear Algebra Library.

#![allow(clippy::needless_range_loop)]

pub mod coo;
pub mod csc;
pub mod csr;
pub mod dok;
pub mod scalar;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dok::DokMatrix;
