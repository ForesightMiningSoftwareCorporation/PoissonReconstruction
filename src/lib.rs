/*!
Rust implementation of the [Screened poisson reconstruction](https://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf)
by Kazhdan and Hoppe.
*/

#![allow(clippy::type_complexity, clippy::too_many_arguments)]
#![warn(missing_docs)]

/// Floating-point type used by this library.
pub type Real = f64;

extern crate nalgebra as na;
extern crate parry3d_f64 as parry;

pub use self::poisson::{PoissonReconstruction};

mod conjugate_gradient;
mod hgrid;
mod poisson;
mod poisson_layer;
mod poisson_vector_field;
mod polynomial;
pub mod marching_cubes;
