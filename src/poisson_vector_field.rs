use crate::poisson_layer::PoissonLayer;
use crate::polynomial::TriQuadraticBspline;
use crate::{poisson, Real};
use itertools::multizip;
use na::{vector, DVector, Point3, Vector3};
use parry::bounding_volume::Aabb;
use rayon::prelude::*;

const CORNERS: [Vector3<i64>; 8] = [
    vector![0, 0, 0],
    vector![1, 0, 0],
    vector![1, 1, 0],
    vector![0, 1, 0],
    vector![0, 0, 1],
    vector![1, 0, 1],
    vector![1, 1, 1],
    vector![0, 1, 1],
];

fn trilinear_coefficients(bcoords: Vector3<Real>) -> [Real; 8] {
    let map = |vals: Vector3<i64>| {
        vals.zip_map(&bcoords, |v, b| if v == 0 { 1.0 - b } else { b })
            .product()
    };

    [
        map(CORNERS[0]),
        map(CORNERS[1]),
        map(CORNERS[2]),
        map(CORNERS[3]),
        map(CORNERS[4]),
        map(CORNERS[5]),
        map(CORNERS[6]),
        map(CORNERS[7]),
    ]
}

pub struct PoissonVectorField {
    pub(crate) densities: Vec<Real>,
    layers_normals: Vec<Vec<Vector3<Real>>>,
}

impl PoissonVectorField {
    pub fn new(
        layers: &[PoissonLayer],
        points: &[Point3<Real>],
        normals: &[Vector3<Real>],
        density_estimation_depth: usize,
    ) -> Self {
        // Compute sample densities.
        let mut densities = vec![1.0; points.len()];
        let density_layer = &layers[density_estimation_depth];
        let mut splat_values = vec![0.0; density_layer.ordered_nodes.len()];

        for pt in points {
            let half_width = Vector3::repeat(density_layer.cell_width() / 2.0);
            // Subtract half-width so the ref_node is the bottom-left node of the trilinear interpolation.
            let ref_node = density_layer.grid.key(&(pt - half_width));

            // Barycentric coordinates of the points for trilinear interpolation.
            let cell_origin = density_layer.grid.cell_center(&ref_node);
            let bcoords = (pt - cell_origin) / density_layer.cell_width();
            let coeffs = trilinear_coefficients(bcoords);

            for (corner_shift, coeff) in CORNERS.iter().zip(coeffs.iter()) {
                let node = ref_node + corner_shift;
                let id = density_layer.grid_node_idx[&node];
                splat_values[id] += *coeff;
            }
        }

        for (pt, weight) in points.iter().zip(densities.iter_mut()) {
            *weight = poisson::eval_triquadratic(
                pt,
                &density_layer.grid,
                &density_layer.grid_node_idx,
                &splat_values,
            );
        }

        let avg_density = densities.iter().copied().sum::<Real>() / (points.len() as Real);
        let samples_depths: Vec<_> = densities
            .iter()
            .map(|d| {
                (((layers.len() - 1) as Real + (*d / avg_density).log(4.0))
                    .round()
                    .max(0.0) as usize)
                    .min(layers.len() - 1)
            })
            .collect();

        let mut layers_normals = vec![];

        for (layer_id, layer) in layers.iter().enumerate() {
            // Splat the normals into the grid.
            let mut grid_normals = vec![Vector3::zeros(); layer.ordered_nodes.len()];

            for (pt, n, w, depth) in multizip((points, normals, &densities, &samples_depths)) {
                if *depth == layer_id {
                    let half_width = Vector3::repeat(layer.grid.cell_width() / 2.0);
                    let ref_node = layer.grid.key(&(pt - half_width));

                    // Barycentric coordinates of the points for trilinear interpolation.
                    let cell_origin = layer.grid.cell_center(&ref_node);
                    let bcoords = (pt - cell_origin) / layer.grid.cell_width();
                    let coeffs = trilinear_coefficients(bcoords);

                    for (corner_shift, coeff) in CORNERS.iter().zip(coeffs.iter()) {
                        let node = ref_node + corner_shift;
                        let id = layer.grid_node_idx[&node];
                        grid_normals[id] += *n * *coeff / *w;
                    }
                }
            }

            layers_normals.push(grid_normals);
        }

        Self {
            densities,
            layers_normals,
        }
    }

    pub fn build_rhs(
        &self,
        layers: &[PoissonLayer],
        curr_layer_id: usize,
        rhs: &mut DVector<Real>,
    ) {
        let curr_layer = &layers[curr_layer_id];

        rhs.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(rhs_id, rhs)| {
                let curr_node = curr_layer.ordered_nodes[rhs_id];
                let curr_node_center = curr_layer.grid.cell_center(&curr_node);

                for (other_layer_id, other_layer) in layers.iter().enumerate() {
                    let aabb = Aabb::from_half_extents(
                        curr_node_center,
                        Vector3::repeat(
                            curr_layer.cell_width() * 1.5 + other_layer.cell_width() * 1.5,
                        ),
                    );

                    for (other_node, _) in other_layer
                        .grid
                        .cells_intersecting_aabb(&aabb.mins, &aabb.maxs)
                    {
                        let other_node_id = other_layer.grid_node_idx[&other_node];
                        let normal = self.layers_normals[other_layer_id][other_node_id];

                        if normal != Vector3::zeros() {
                            let other_node_center = other_layer.grid.cell_center(&other_node);
                            let poly1 = TriQuadraticBspline::new(
                                other_node_center,
                                other_layer.cell_width(),
                            );
                            let poly2 =
                                TriQuadraticBspline::new(curr_node_center, curr_layer.cell_width());
                            let coeff = poly1.grad_grad(poly2, false, true);
                            *rhs += normal.dot(&coeff);
                        }
                    }
                }
            });
    }

    pub fn area_approximation(&self) -> Real {
        self.densities.iter().map(|d| 1.0 / *d).sum()
    }
}
