use super::datatypes;
use std::ops::{Add, Sub};

impl Add for datatypes::Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.idx() != rhs.idx() {
            panic!("dimensions of tensor do not match")
        }
        let mut result = datatypes::Tensor::new(self.idx().clone());
        // we are able to use little hack with accessing raw data directly
        for i in 0..self.raw_data.len() {
            result.raw_data[i] = self.raw_data[i] + rhs.raw_data[i];
        }
        result
    }
}

impl Sub for datatypes::Tensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.idx() != rhs.idx() {
            panic!("dimensions of tensor do not match")
        }
        let mut result = datatypes::Tensor::new(self.idx().clone());
        // we are able to use little hack with accessing raw data directly
        for i in 0..self.raw_data.len() {
            result.raw_data[i] = self.raw_data[i] - rhs.raw_data[i];
        }
        result
    }
}

// https://www.mathworks.com/help/matlab/ref/tensorprod.html
impl datatypes::Tensor {
    pub fn inner_product(&self, other: &Self) -> f64 {
        assert!(self.idx() == other.idx());
        let mut res = 0.0;
        // we could implement inner product via for loops sum
        // but this seems to be unnecessary since we have access to the raw_data of both tensors
        for i in 0..self.raw_data.len() {
            res += self.raw_data[i] * other.raw_data[i];
        }
        res
    }

    pub fn outer_product(&self, other: &Self) -> Self {
        let new_dims = [self.idx().dims.as_slice(), other.idx().dims.as_slice()].concat();
        // for outer product unfortunately we will not be able to use this hack :(
        let new_shape = datatypes::Idx::new(&new_dims);

        let mut res = datatypes::Tensor::new(new_shape.clone());

        let mut r_idx = datatypes::Idx::new(&vec![0 as usize; new_dims.len()]);
        loop {
            // iterating over each of dimensions
            // until rank(A) it is i-s, then j-s
            // (i0, ..., i_rank_A-1) (j0, ..., j_rank_B-1)
            let mut idx = r_idx.dims.len() - 1;
            loop {
                let a_idx = datatypes::Idx::new(&r_idx.dims[0..self.rank()].to_vec());
                let b_idx = datatypes::Idx::new(&r_idx.dims[self.rank()..].to_vec());
                res[r_idx.clone()] = self[a_idx] * other[b_idx];
                if r_idx.dims[idx] < new_dims[idx] - 1 {
                    r_idx.dims[idx] += 1;
                    break;
                } else {
                    r_idx.dims[idx] = 0;
                    if idx == 0 {
                        return res;
                    }
                    idx -= 1
                }
            }
        }
    }

    pub fn tensor_transpose(&self) -> Self {
        let new_dims: Vec<usize> = self.idx().dims.iter().copied().rev().collect();
        let new_shape = datatypes::Idx::new(&new_dims);
        let mut res = datatypes::Tensor::new(new_shape);

        let mut r_idx = datatypes::Idx::new(&vec![0 as usize; new_dims.len()]);
        let mut done = false;
        while !done {
            let mut idx = 0;
            loop {
                let self_idx =
                    datatypes::Idx::new(&r_idx.clone().dims.iter().copied().rev().collect());
                res[r_idx.clone()] = self[self_idx];
                // if (x,y,z) is in match for T then (z, y, x) should be for T^T
                r_idx.dims[idx] += 1;
                if r_idx.dims[idx] < new_dims[idx] {
                    break;
                }
                r_idx.dims[idx] = 0;
                idx += 1;
                if idx >= r_idx.len() {
                    done = true;
                    break;
                }
            }
        }
        res
    }

    pub fn scalar_multiply(&self, scalar: f64) -> Self {
        let mut tensor = self.clone();

        for i in 0..self.raw_data.len() {
            tensor.raw_data[i] *= scalar;
        }

        tensor
    }
}

#[cfg(test)]
mod tests {
    use crate::core::datatypes;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn create_tensor() {
        let idx = datatypes::Idx::new(&vec![2, 3]);
        let mut tensor = datatypes::Tensor::new(idx);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.shape(), vec![2, 3]);
        assert_approx_eq!(tensor[datatypes::Idx::new(&vec![0, 0])], 0.0);
        assert_approx_eq!(tensor[datatypes::Idx::new(&vec![0, 1])], 0.0);
        assert_approx_eq!(tensor[datatypes::Idx::new(&vec![1, 0])], 0.0);
        assert_approx_eq!(tensor[datatypes::Idx::new(&vec![1, 1])], 0.0);
        tensor[datatypes::Idx::new(&vec![0, 1])] = 3.0;
        assert_approx_eq!(tensor[datatypes::Idx::new(&vec![0, 1])], 3.0);
        let tensor2 = tensor.clone();
        assert_eq!(tensor, tensor2);
        let tensor3 = tensor2.tensor_transpose();
        println!("{:?}", tensor3);
        assert_approx_eq!(tensor3[datatypes::Idx::new(&vec![0, 0])], 0.0);
        assert_approx_eq!(tensor3[datatypes::Idx::new(&vec![0, 1])], 0.0);
        assert_approx_eq!(tensor3[datatypes::Idx::new(&vec![1, 0])], 3.0);
        assert_approx_eq!(tensor3[datatypes::Idx::new(&vec![1, 1])], 0.0);
    }
}
