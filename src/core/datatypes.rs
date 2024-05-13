#[derive(Clone, PartialEq, Debug)]
pub struct Idx {
    pub(super) dims: Vec<usize>,
}

impl Idx {
    pub fn new(dims: &Vec<usize>) -> Self {
        Idx { dims: dims.clone() }
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct Tensor {
    idx: Idx,
    pub(super) raw_data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Idx) -> Self {
        let total_len: usize = shape.dims.iter().product();
        Tensor {
            idx: shape,
            raw_data: vec![0.0; total_len],
        }
    }

    fn new_with_raw_data(shape: Idx, raw_data: Vec<f64>) -> Self {
        Tensor {
            idx: shape,
            raw_data: raw_data,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.idx.dims.clone()
    }

    pub fn rank(&self) -> usize {
        self.idx.dims.len()
    }

    pub(super) fn idx(&self) -> Idx {
        self.idx.clone()
    }
}

pub struct TensorBuilder {
    pub len: usize,
    pub shape: Vec<usize>,
    pub raw_data: Vec<f64>,
}

impl TensorBuilder {
    pub fn new() -> Self {
        TensorBuilder {
            len: 0,
            shape: Vec::new(),
            raw_data: Vec::new(),
        }
    }

    pub fn build(self) -> Tensor {
        return Tensor::new_with_raw_data(Idx::new(&self.shape), self.raw_data);
    }
}

// NOTE: this is prototype
pub enum Either {
    Scalar(f64),
    Tensor(Tensor),
}
