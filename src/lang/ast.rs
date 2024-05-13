#[derive(Debug, Clone)]
pub enum Expr {
    Identifier(String),
    Literal(f64),
    FunctionCall {
        name: String,
        args: Vec<Expr>,
    },
    Tensor(Vec<Expr>),
    BinaryOp {
        op: String,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Assignment {
        name: String,
        expr: Box<Expr>,
    },
}
