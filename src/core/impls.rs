use super::datatypes;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

impl datatypes::Tensor {
    fn flatten_index(&self, index: &datatypes::Idx) -> usize {
        let under: Vec<usize> = index.dims.iter().copied().rev().collect();
        // index will we representing (x, y, z, ...)
        let shape: Vec<usize> = self.idx().dims.iter().copied().rev().collect();
        let mut idx = under[0];
        let mut prod = 1;
        for i in 1..under.len() {
            if shape[i] <= under[i] {
                panic!("Index out of bounds")
            }
            prod *= shape[i - 1];
            idx += under[i] * prod;
        }
        return idx;
    }
}

impl Index<datatypes::Idx> for datatypes::Tensor {
    type Output = f64;

    fn index(&self, index: datatypes::Idx) -> &Self::Output {
        if self.rank() != index.dims.len() {
            panic!("Index length does not match tensor rank")
        }
        let idx = self.flatten_index(&index);
        &self.raw_data[idx]
    }
}

impl IndexMut<datatypes::Idx> for datatypes::Tensor {
    fn index_mut(&mut self, index: datatypes::Idx) -> &mut Self::Output {
        if self.rank() != index.dims.len() {
            panic!("Index length does not match tensor rank")
        }
        let idx = self.flatten_index(&index);
        &mut self.raw_data[idx]
    }
}

impl Display for datatypes::Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut offset: usize = 0;
        let out = self.format(0, &mut offset);
        write!(f, "{}", out)
    }
}


impl datatypes::Tensor {
    
    fn format(&self, dim_idx: usize, offset: &mut usize) -> String {
        let mut out = String::new();
        out += "[";
        if dim_idx == self.rank()-1 {
            for i in 0..self.idx().dims[dim_idx] {
                if i > 0 {
                    out += ", "
                }
                out += &format!("{}", self.raw_data[*offset+i])
            }
            *offset += self.idx().dims[dim_idx];
        } else {
            for i in 0..self.idx().dims[dim_idx] {
                if i > 0 {
                    out += ", ";
                }    
                out += &self.format(dim_idx+1, offset);
            }  
        }
        out += "]"; 
        out
    }
}