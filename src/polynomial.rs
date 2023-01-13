use crate::Real;
use na::{Point3, Vector3};
use std::ops::{Add, Div, Mul, Neg};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TriQuadraticBspline {
    center: Point3<Real>,
    width: Real,
}

impl TriQuadraticBspline {
    pub fn new(center: Point3<Real>, width: Real) -> Self {
        Self { center, width }
    }

    pub fn eval(&self, pt: Point3<Real>) -> Real {
        let mut result = 1.0;

        for i in 0..3 {
            result *= eval_bspline(pt[i], self.center[i], self.width)
        }

        result
    }

    pub fn grad_grad(self, rhs: Self, diff1: bool, diff2: bool) -> Vector3<Real> {
        let dcenter = rhs.center - self.center;
        let poly1 = bspline::<6>(0.0, self.width);
        let poly_diff1 = if diff1 {
            [
                poly1[0].derivative(),
                poly1[1].derivative(),
                poly1[2].derivative(),
            ]
        } else {
            poly1
        };

        let mut result_int = [0.0; 3];
        let mut result_diff_diff_int = [0.0; 3];

        for dim in 0..3 {
            if dcenter[dim].abs() >= (self.width + rhs.width) * 1.5 {
                return Vector3::zeros(); // The splines don’t overlap along this dimension.
            }

            // We have to check the splines domain pieces to multiply together
            // the correct polynomials.
            let sub1 = [
                -1.5 * self.width,
                -0.5 * self.width,
                0.5 * self.width,
                1.5 * self.width,
            ];
            let sub2 = [
                dcenter[dim] - 1.5 * rhs.width,
                dcenter[dim] - 0.5 * rhs.width,
                dcenter[dim] + 0.5 * rhs.width,
                dcenter[dim] + 1.5 * rhs.width,
            ];

            let poly2 = bspline::<6>(dcenter[dim], rhs.width);
            let poly_diff2 = if diff2 {
                [
                    poly2[0].derivative(),
                    poly2[1].derivative(),
                    poly2[2].derivative(),
                ]
            } else {
                poly2
            };

            // Compute the 9 potential interval intersections.
            for i in 0..3 {
                for j in 0..3 {
                    let start = sub1[i].max(sub2[j]);
                    let end = sub1[i + 1].min(sub2[j + 1]);
                    if end > start {
                        let primitive = (poly1[i] * poly2[j]).primitive();
                        result_int[dim] += primitive.eval(end) - primitive.eval(start);

                        let primitive_diff_diff = (poly_diff1[i] * poly_diff2[j]).primitive();
                        result_diff_diff_int[dim] +=
                            primitive_diff_diff.eval(end) - primitive_diff_diff.eval(start);
                    }
                }
            }
        }

        Vector3::new(
            result_diff_diff_int[0] * result_int[1] * result_int[2],
            result_int[0] * result_diff_diff_int[1] * result_int[2],
            result_int[0] * result_int[1] * result_diff_diff_int[2],
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PoissonQuadraticBsplineCoeffs {
    pub laplacian: [[[Real; 5]; 5]; 5],
    pub normal_div: [[[[Real; 5]; 5]; 5]; 3],
}

impl Default for PoissonQuadraticBsplineCoeffs {
    fn default() -> Self {
        Self {
            laplacian: [[[0.0; 5]; 5]; 5],
            normal_div: [[[[0.0; 5]; 5]; 5]; 3],
        }
    }
}

fn bspline03<const DEG: usize>() -> [Polynomial<DEG>; 3] {
    [
        Polynomial::<DEG>::quadratic(0.0, 0.0, 0.5), // x in [0, 1)
        Polynomial::<DEG>::quadratic(-1.5, 3.0, -1.0), // x in [1, 2)
        Polynomial::<DEG>::quadratic(4.5, -3.0, 0.5), // x in [2, 3)
    ]
}

fn bspline<const DEG: usize>(origin: Real, width: Real) -> [Polynomial<DEG>; 3] {
    let b = bspline03::<DEG>();

    [
        b[0].scale_shift(origin - 1.5 * width, width) / width,
        b[1].scale_shift(origin - 1.5 * width, width) / width,
        b[2].scale_shift(origin - 1.5 * width, width) / width,
    ]
}

pub fn eval_bspline(x: Real, origin: Real, width: Real) -> Real {
    // Bring the value between [0, 3)
    let val = (x - origin) / width + 1.5;
    let [b0, b1, b2] = bspline03::<3>();

    if val < 0.0 {
        0.0
    } else if val < 1.0 {
        b0.eval(val) / width
    } else if val < 2.0 {
        b1.eval(val) / width
    } else if val < 3.0 {
        b2.eval(val) / width
    } else {
        0.0
    }
}

pub fn eval_bspline_diff(x: Real, origin: Real, width: Real) -> Real {
    // Bring the value between [0, 3)
    let val = (x - origin) / width + 1.5;
    let [b0, b1, b2] = bspline03::<3>();

    if val < 0.0 {
        0.0
    } else if val < 1.0 {
        b0.derivative().eval(val) / width
    } else if val < 2.0 {
        b1.derivative().eval(val) / width
    } else if val < 3.0 {
        b2.derivative().eval(val) / width
    } else {
        0.0
    }
}

pub fn compute_quadratic_bspline_convolution_coeffs(width: Real) -> PoissonQuadraticBsplineCoeffs {
    let [mut b0, mut b1, mut b2] = bspline03::<6>();

    // Center and normalize each section of the b-spline.
    b0 = b0.scale_shift(0.0, width) / width; // x in [0, w)
    b1 = b1.scale_shift(-width, width) / width; // x in [0, w)
    b2 = b2.scale_shift(-2.0 * width, width) / width; // x in [0, w)
    let bdiff0 = b0.derivative();
    let bdiff1 = b1.derivative();
    let bdiff2 = b2.derivative();

    let primitives = [
        (b2 * b0).primitive(),
        (b1 * b0).primitive() + (b2 * b1).primitive(),
        (b0 * b0).primitive() + (b1 * b1).primitive() + (b2 * b2).primitive(),
        (b1 * b0).primitive() + (b2 * b1).primitive(),
        (b2 * b0).primitive(),
    ];
    let integrals = [
        primitives[0].eval(width) - primitives[0].eval(0.0),
        primitives[1].eval(width) - primitives[1].eval(0.0),
        primitives[2].eval(width) - primitives[2].eval(0.0),
        primitives[3].eval(width) - primitives[3].eval(0.0),
        primitives[4].eval(width) - primitives[4].eval(0.0),
    ];

    let primitives_diff = [
        (bdiff2 * b0).primitive(),
        (bdiff1 * b0).primitive() + (bdiff2 * b1).primitive(),
        (b0 * bdiff0).primitive() + (b1 * bdiff1).primitive() + (b2 * bdiff2).primitive(),
        (b1 * bdiff0).primitive() + (b2 * bdiff1).primitive(),
        (b2 * bdiff0).primitive(),
    ];
    let integrals_diff = [
        primitives_diff[0].eval(width) - primitives_diff[0].eval(0.0),
        primitives_diff[1].eval(width) - primitives_diff[1].eval(0.0),
        primitives_diff[2].eval(width) - primitives_diff[2].eval(0.0),
        primitives_diff[3].eval(width) - primitives_diff[3].eval(0.0),
        primitives_diff[4].eval(width) - primitives_diff[4].eval(0.0),
    ];

    let primitives_diff_diff = [
        (bdiff2 * bdiff0).primitive(),
        ((bdiff1 * bdiff0).primitive() + (bdiff2 * bdiff1).primitive()),
        ((bdiff0 * bdiff0).primitive()
            + (bdiff1 * bdiff1).primitive()
            + (bdiff2 * bdiff2).primitive()),
        ((bdiff1 * bdiff0).primitive() + (bdiff2 * bdiff1).primitive()),
        ((bdiff2 * bdiff0).primitive()),
    ];
    let integrals_diff_diff = [
        primitives_diff_diff[0].eval(width) - primitives_diff_diff[0].eval(0.0),
        primitives_diff_diff[1].eval(width) - primitives_diff_diff[1].eval(0.0),
        primitives_diff_diff[2].eval(width) - primitives_diff_diff[2].eval(0.0),
        primitives_diff_diff[3].eval(width) - primitives_diff_diff[3].eval(0.0),
        primitives_diff_diff[4].eval(width) - primitives_diff_diff[4].eval(0.0),
    ];

    let mut result = PoissonQuadraticBsplineCoeffs::default();

    for i in -2i32..=2 {
        for j in -2i32..=2 {
            for k in -2i32..=2 {
                let ia = (i + 2) as usize;
                let ja = (j + 2) as usize;
                let ka = (k + 2) as usize;

                result.laplacian[ia][ja][ka] =
                    integrals_diff_diff[ia] * integrals[ja] * integrals[ka]
                        + integrals[ia] * integrals_diff_diff[ja] * integrals[ka]
                        + integrals[ia] * integrals[ja] * integrals_diff_diff[ka];
                result.normal_div[0][ia][ja][ka] =
                    integrals_diff[ia] * integrals[ja] * integrals[ka];
                result.normal_div[1][ia][ja][ka] =
                    integrals[ia] * integrals_diff[ja] * integrals[ka];
                result.normal_div[2][ia][ja][ka] =
                    integrals[ia] * integrals[ja] * integrals_diff[ka];
            }
        }
    }

    result
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Polynomial<const N: usize> {
    pub coeffs: [Real; N],
}

impl<const N: usize> Default for Polynomial<N> {
    fn default() -> Self {
        Self { coeffs: [0.0; N] }
    }
}

impl<const N: usize> Polynomial<N> {
    pub fn eval(&self, x: Real) -> Real {
        let mut result = self.coeffs[N - 1];

        for i in (0..N - 1).rev() {
            result = result * x + self.coeffs[i];
        }

        result
    }

    #[must_use]
    pub fn quadratic(cst: Real, x: Real, xx: Real) -> Self {
        let mut coeffs = [0.0; N];
        coeffs[0] = cst;
        coeffs[1] = x;
        coeffs[2] = xx;
        Self { coeffs }
    }

    #[must_use]
    pub fn derivative(mut self) -> Self {
        for i in 0..N - 1 {
            self.coeffs[i] = self.coeffs[i + 1] * (i as Real + 1.0);
        }
        self.coeffs[N - 1] = 0.0;
        self
    }

    #[must_use]
    pub fn primitive(mut self) -> Self {
        assert_eq!(
            self.coeffs[N - 1],
            0.0,
            "Integration coefficient overflow. Increase the polynomial degree."
        );
        for i in (1..N).rev() {
            self.coeffs[i] = self.coeffs[i - 1] / (i as Real);
        }
        self.coeffs[0] = 0.0;
        self
    }

    // For a polynomial up to degree 2, this computes the polynomial
    // representation of P(X) = P((x - center) / width)
    #[must_use]
    pub fn scale_shift(self, center: Real, width: Real) -> Self {
        for k in 3..N {
            assert_eq!(
                self.coeffs[k], 0.0,
                "Only implemented for polynomials with degrees up to 2."
            );
        }

        let a = self.coeffs[0];
        let b = self.coeffs[1];
        let c = self.coeffs[2];
        let w = width;
        let ww = w * w;

        let mut result = Self::default();
        result.coeffs[0] = a - center * b / w + c * center * center / ww;
        result.coeffs[1] = b / w - 2.0 * c * center / ww;
        result.coeffs[2] = c / ww;
        result
    }
}

impl<const N: usize> Neg for Polynomial<N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for i in 0..N {
            self.coeffs[i] = -self.coeffs[i];
        }
        self
    }
}

impl<const N: usize> Div<Real> for Polynomial<N> {
    type Output = Self;
    fn div(mut self, rhs: Real) -> Self {
        for i in 0..N {
            self.coeffs[i] /= rhs;
        }
        self
    }
}

impl<const N: usize> Mul<Polynomial<N>> for Polynomial<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut result = Self::default();
        for i in 0..N {
            for j in 0..N {
                let val = self.coeffs[i] * rhs.coeffs[j];
                if j + i >= N {
                    assert_eq!(
                        val, 0.0,
                        "The result of the product must have a degree smaller than N"
                    );
                } else {
                    result.coeffs[j + i] += self.coeffs[i] * rhs.coeffs[j];
                }
            }
        }
        result
    }
}

impl<const N: usize> Add<Polynomial<N>> for Polynomial<N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
        self
    }
}

#[cfg(test)]
mod test {
    use crate::polynomial::Polynomial;

    #[test]
    fn poly_eval() {
        let poly = Polynomial {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0],
        };
        assert_eq!(
            poly.eval(2.0),
            1.0 + 2.0 * 2.0 + 3.0 * 4.0 + 4.0 * 8.0 + 5.0 * 16.0
        );
    }

    #[test]
    fn poly_add() {
        let poly1 = Polynomial {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let poly2 = Polynomial {
            coeffs: [10.0, 20.0, 30.0, 40.0, 50.0],
        };
        let expected = Polynomial {
            coeffs: [11.0, 22.0, 33.0, 44.0, 55.0],
        };
        assert_eq!(poly1 + poly2, expected);
    }

    #[test]
    fn poly_mul() {
        let poly1 = Polynomial {
            coeffs: [1.0, 2.0, 3.0, 0.0, 0.0],
        };
        let poly2 = Polynomial {
            coeffs: [10.0, 20.0, 30.0, 0.0, 0.0],
        };
        let expected = Polynomial {
            coeffs: [10.0, 40.0, 100.0, 120.0, 90.0],
        };
        assert_eq!(poly1 * poly2, expected);
    }

    #[test]
    fn poly_diff() {
        let poly = Polynomial {
            coeffs: [1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let expected = Polynomial {
            coeffs: [2.0, 6.0, 12.0, 20.0, 0.0],
        };
        assert_eq!(poly.derivative(), expected);
    }

    #[test]
    fn poly_primitive() {
        let poly = Polynomial {
            coeffs: [1.0, 2.0, 3.0, 4.0, 0.0],
        };
        let expected = Polynomial {
            coeffs: [0.0, 1.0, 1.0, 1.0, 1.0],
        };
        assert_eq!(poly.primitive(), expected);
        assert_eq!(poly.primitive().derivative(), poly);
    }

    #[test]
    fn scale_shift() {
        let shift = 0.5;
        let width = 2.5;
        let poly = Polynomial {
            coeffs: [10.0, 20.0, 30.0, 0.0, 0.0],
        };
        let poly_scale_shifted = poly.scale_shift(shift, width);
        assert_eq!(
            poly.eval((11.0 - shift) / width),
            poly_scale_shifted.eval(11.0)
        );
        assert_eq!(poly.eval(0.0), poly_scale_shifted.eval(shift));
        assert!((poly.eval(-shift / width) - poly_scale_shifted.eval(0.0)).abs() < 1.0e-8);
    }
}
