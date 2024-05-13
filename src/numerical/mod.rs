use crate::core::datatypes::{Tensor, Idx};
use nalgebra::{DMatrix, Dynamic, Matrix, U1, U2, U3};

// TODO: this module to be implemented

impl Tensor {
	
	fn flatten_mode_m(&self, m: usize) -> Vec<Vec<f64>> {
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
		let mut index = vec![0 as usize; dims.len()];
		// TODO: implement
		for row in 0..mode_size {
			let mut dim_idx = 0;
			let mut sum_cols = 0 as usize;
			for column in 0..row_size {
				for i in 0..dims.len() {
					if i == m {
						index[m] = row;
					} else {
						if false {
							index[m] = dims[i];
						} else {
							index[m] = column - sum_cols;
						}
					}
				}				
				let idx = Idx::new(&index);
			}
		}
		
		return matrix
	}
	
	pub fn svd(matrix: Vec<Vec<f64>>) {
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
	
		// Extract U, S, and V matrices
		let u = svd.u.unwrap();
		let s = svd.singular_values;
		let v = svd.v_t.unwrap();
	
	}
	
	
	// implementation of HOSVD
	pub fn hosvd(&self) {
		for m in 0..self.rank() {
			let a_m = self.flatten_mode_m(m);
			// svd(a_m)
			
			// construct mode-m flattening A_m
			// compute (compact) SVD A_m = U_m S_m V_m^T and store left singular vectors
		}
		// compute core tensor S with multilinear multiplication S = A x U_1^H x ... x U_m^h
	}
}