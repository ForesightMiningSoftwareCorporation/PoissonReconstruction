use crate::Real;
use na::DVector;
use nalgebra_sparse::CscMatrix;

pub fn solve_conjugate_gradient(a: &CscMatrix<Real>, b: &mut DVector<Real>, niters: usize) {
    let mut r = &*b - a * &*b;
    let mut p = r.clone();
    let mut prev_rr = r.dot(&r);

    for _ in 0..niters {
        let ap = a * &p; // TODO: avoid the allocation.
        let alpha = r.dot(&r) / p.dot(&ap);
        b.axpy(alpha, &p, 1.0);
        r.axpy(-alpha, &ap, 1.0);
        let rr = r.dot(&r);
        let beta = rr / prev_rr;
        prev_rr = rr;
        p.axpy(1.0, &r, beta);
    }
}
