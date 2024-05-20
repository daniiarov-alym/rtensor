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

impl datatypes::Tensor {
    fn format_array(&self, level: usize) -> String {
        let elements = self.idx().dims[self.idx().dims.len() - 1];
        let mut out = String::new();
        let mut first = true;
        out += "[";
        for i in 0..elements {
            if !first {
                out += ", "
            }
            first = false;
            out += &format!("{}", self.raw_data[level * elements + i])
        }
        out += "]";
        out
    }

    fn format_impl_inner(&self, idx: usize, dim: usize) -> String {
        let mut out = String::new();
        if dim == self.idx().dims.len() - 1 {
            out = self.format_array(idx);
            return out;
        }
        let mut first = true;
        let dims = self.idx().dims.clone();
        out += "[";
        for _ in 0..dims[dim] {
            if !first {
                out += ", ";
            }
            first = false;
            out += &self.format_impl_inner(idx, dim + 1);
        }
        out += "]";
        return out;
    }
// FIXME: error A=[[[1,2],[3,4]],[[5,6],[7,8]]] is represented as [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]
    fn format_impl(&self) -> String {
        let dims = self.idx().dims.clone();
        let mut out = String::new();

        if dims.len() == 1 {
            out = self.format_array(0);
            return out;
        }
        let mut first = true;
        for i in 0..dims[0] {
            if !first {
                out += ", ";
            }
            first = false;
            out += &self.format_impl_inner(i, 1);
        }
        out
    }
}

impl Display for datatypes::Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();

        if self.idx().dims.len() == 1 {
            out = self.format_array(0)
        } else {
            out += "[";
            out += &self.format_impl();
            out += "]";
        }
        write!(f, "{}", out)
    }
}
