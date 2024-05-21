use std::vec;

use crate::core::datatypes::Tensor;
use nalgebra::DMatrix;

impl Tensor {
    // NOTE: seems like here is an error
    fn mode_m_matrixizing(&self, m: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.num_cols(m)]; self.shape()[m]];
        let mut multi_index = vec![0; self.shape().len()];

        for i in 0..self.data().len() {
            let (j, k) = self.calculate_indices(i, m, &mut multi_index);
            result[j][k] = self.data()[i];
        }

        result
    }

    // Calculate the number of columns for the resulting matrix
    fn num_cols(&self, m: usize) -> usize {
        self.shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != m)
            .map(|(_, &dim)| dim)
            .product()
    }

    // Calculate the linear indices for the mode-m matrix
    fn calculate_indices(
        &self,
        flat_index: usize,
        m: usize,
        multi_index: &mut Vec<usize>,
    ) -> (usize, usize) {
        self.flat_to_multi_index(flat_index, multi_index);

        let j = multi_index[m];

        let k = multi_index
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != m)
            .fold(0, |acc, (i, &idx)| acc * self.shape()[i] + idx);

        (j, k)
    }

    fn flat_to_multi_index(&self, flat_index: usize, multi_index: &mut Vec<usize>) {
        let dims_product: Vec<usize> = self
            .shape()
            .iter()
            .rev()
            .scan(1, |state, &x| {
                let res = *state;
                *state *= x;
                Some(res)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        for i in 0..self.shape().len() {
            multi_index[i] = (flat_index / dims_product[i]) % self.shape()[i];
        }
    }

    /*fn flatten_mode_m(&self, m: usize) -> Vec<Vec<f64>> {
        // R[j, k] = self[i1, ..., im, ..., iM]
        // j = i_m, k = 1 + sum((i_n-1)*prod(Il))
        let dims = self.shape();
        let mode_size = dims[m];
        let mut row_size: usize = 1;
        for i in 0..dims.len() {
            if i != m {
                row_size *= dims[i];
            }
        }
        let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; row_size]; mode_size];
        // TODO: implement

        // j = i_m
        // k = 1 + sum((i_n-1)prod(n-1))

        return matrix
    }*/

    pub fn svd(&self, matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut mat = DMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                mat[(i, j)] = matrix[i][j];
            }
        }

        // Compute SVD
        let svd = mat.svd(true, true); // Compute left and right singular vectors

        // Extract U matrix
        let u = svd.u.unwrap();

        let u_t = u.transpose();

        let mut res: Vec<Vec<f64>> = vec![vec![0.0; u_t.ncols()]; u_t.nrows()];
        for i in 0..u_t.nrows() {
            for j in 0..u_t.ncols() {
                res[i][j] = u_t[(i, j)]
            }
        }
        return res;
    }

    // implementation of HOSVD
    pub fn hosvd(&self) {
        let mut singular_vectors: Vec<Vec<Vec<f64>>> = Vec::new();
        for m in 0..self.rank() {
            let a_m = self.mode_m_matrixizing(m);
            let u_m = self.svd(a_m);
            singular_vectors.push(u_m)
        }

        unimplemented!()
        // compute core tensor S with multilinear multiplication S = A x U_1^H x ... x U_m^h
    }
}
