// This is a Hierarchical Grid taken from Salva.
// We should probably move this to Parry.
#![allow(dead_code)]

use fnv::FnvHasher;

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::Real;
use na::{Point3, Vector3};

#[derive(Copy, Clone, Debug)]
pub struct DeterministicState;

impl Default for DeterministicState {
    fn default() -> Self {
        DeterministicState
    }
}

impl BuildHasher for DeterministicState {
    type Hasher = FnvHasher;

    fn build_hasher(&self) -> FnvHasher {
        FnvHasher::with_key(1820)
    }
}

/// A grid based on spacial hashing.
#[derive(PartialEq, Debug, Clone)]
#[cfg_attr(
    feature = "serde-serialize",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct HGrid<T> {
    cells: HashMap<Point3<i64>, Vec<T>, DeterministicState>,
    origin: Point3<Real>,
    cell_width: Real,
}

impl<T> HGrid<T> {
    /// Initialize a grid where each cell has the width `cell_width`.
    pub fn new(origin: Point3<Real>, cell_width: Real) -> Self {
        Self {
            cells: HashMap::with_hasher(DeterministicState),
            origin,
            cell_width,
        }
    }

    /// The width of a cell of this spacial grid.
    pub fn cell_width(&self) -> Real {
        self.cell_width
    }

    /// The origin of this grid.
    pub fn origin(&self) -> &Point3<Real> {
        &self.origin
    }

    fn unquantify(value: i64, cell_width: Real) -> Real {
        value as Real * cell_width + cell_width / 2.0
    }

    fn quantify(value: Real, cell_width: Real) -> i64 {
        na::try_convert::<Real, f64>((value / cell_width).floor()).unwrap() as i64
    }

    fn quantify_ceil(value: Real, cell_width: Real) -> i64 {
        na::try_convert::<Real, f64>((value / cell_width).ceil()).unwrap() as i64
    }

    /// Computes the logical grid cell containing `point`.
    pub fn key(&self, point: &Point3<Real>) -> Point3<i64> {
        Point3::from((point - self.origin).map(|e| Self::quantify(e, self.cell_width)))
    }

    /// Removes all elements from this grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Inserts the given `element` into the cell containing the given `point`.
    pub fn insert(&mut self, point: &Point3<Real>, element: T) {
        let key = self.key(point);
        self.cells.entry(key).or_insert_with(Vec::new).push(element)
    }

    /// Returns the element attached to the cell containing the given `point`.
    ///
    /// Returns `None` if the cell is empty.
    pub fn cell_containing_point(&self, point: &Point3<Real>) -> Option<&Vec<T>> {
        let key = self.key(point);
        self.cells.get(&key)
    }

    /// An iterator through all the non-empty cells of this grid.
    ///
    /// The returned tuple include the cell indentifier, and the elements attached to this cell.
    pub fn cells(&self) -> impl Iterator<Item = (&Point3<i64>, &Vec<T>)> {
        self.cells.iter()
    }

    /// The underlying hash map of this spacial grid.
    pub fn inner_table(&self) -> &HashMap<Point3<i64>, Vec<T>, DeterministicState> {
        &self.cells
    }

    /// Get the content of the logical cell identified by `key`.
    pub fn cell(&self, key: &Point3<i64>) -> Option<&Vec<T>> {
        self.cells.get(key)
    }

    pub fn cell_center(&self, cell: &Point3<i64>) -> Point3<Real> {
        self.origin + cell.coords.map(|x| Self::unquantify(x, self.cell_width))
    }

    /// An iterator through all the neighbors of the given cell.
    ///
    /// The given cell itself will be yielded by this iterator too.
    pub fn neighbor_cells(
        &self,
        cell: &Point3<i64>,
        radius: Real,
    ) -> impl Iterator<Item = (Point3<i64>, &Vec<T>)> {
        let cells = &self.cells;
        let quantified_radius = Self::quantify_ceil(radius, self.cell_width);

        CellRangeIterator::with_center(*cell, quantified_radius)
            .filter_map(move |cell| cells.get(&cell).map(|c| (cell, c)))
    }

    /// An iterator through all the neighbors of the given cell, including empty cells.
    ///
    /// The given cell itself will be yielded by this iterator too.
    pub fn maybe_neighbor_cells(
        &self,
        cell: &Point3<i64>,
        radius: Real,
    ) -> impl Iterator<Item = (Point3<i64>, Option<&Vec<T>>)> {
        let cells = &self.cells;
        let quantified_radius = Self::quantify_ceil(radius, self.cell_width);
        CellRangeIterator::with_center(*cell, quantified_radius)
            .map(move |cell| (cell, cells.get(&cell)))
    }

    /// An iterator through all the cells intersecting the given Aabb.
    pub fn cells_intersecting_aabb(
        &self,
        mins: &Point3<Real>,
        maxs: &Point3<Real>,
    ) -> impl Iterator<Item = (Point3<i64>, &Vec<T>)> {
        let cells = &self.cells;
        let start = self.key(mins);
        let end = self.key(maxs);

        CellRangeIterator::new(start, end)
            .filter_map(move |cell| cells.get(&cell).map(|c| (cell, c)))
    }

    /// An iterator through all the cells intersecting the given Aabb, including empty cells.
    pub fn maybe_cells_intersecting_aabb(
        &self,
        mins: &Point3<Real>,
        maxs: &Point3<Real>,
    ) -> impl Iterator<Item = (Point3<i64>, Option<&Vec<T>>)> {
        let cells = &self.cells;
        let start = self.key(mins);
        let end = self.key(maxs);

        CellRangeIterator::new(start, end).map(move |cell| (cell, cells.get(&cell)))
    }
}

struct CellRangeIterator {
    start: Point3<i64>,
    end: Point3<i64>,
    curr: Point3<i64>,
    done: bool,
}

impl CellRangeIterator {
    fn new(start: Point3<i64>, end: Point3<i64>) -> Self {
        Self {
            start,
            end,
            curr: start,
            done: false,
        }
    }

    fn with_center(center: Point3<i64>, radius: i64) -> Self {
        let start = center - Vector3::repeat(radius);
        Self {
            start,
            end: center + Vector3::repeat(radius),
            curr: start,
            done: false,
        }
    }
}

impl Iterator for CellRangeIterator {
    type Item = Point3<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.curr == self.end {
            self.done = true;
            Some(self.curr)
        } else {
            let result = self.curr;

            for i in 0..3 {
                self.curr[i] += 1;

                if self.curr[i] > self.end[i] {
                    self.curr[i] = self.start[i];
                } else {
                    break;
                }
            }

            Some(result)
        }
    }
}
