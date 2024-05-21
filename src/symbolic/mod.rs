use crate::core::datatypes;

/*

    symbolic expression is either:
    1. pure symbol
    2. function call with one ore more arguments being symbols
    3. binary operation with one or both operands being symbols
    4. numerical expression involving symbol?

*/
#[derive(Debug, Clone)]
pub enum SymbolicExpr {
    Symbol(String),

    FunctionCall {
        name: String,
        args: Vec<SymbolicExpr>,
    },

    BinaryOp {
        op: String,
        left: Box<SymbolicExpr>,
        right: Box<SymbolicExpr>,
    },
    /*
        // for that was evaluated
        TensorSymbol {
            name: String,
            tensor: datatypes::Tensor,
        },

        // for that was evaluated
        LiteralSymbol {
            name: String,
            literal: f64,
        },
    */
    UnnamedTensor {
        tensor: datatypes::Tensor,
    },

    UnnamedLiteral {
        literal: f64,
    },
}
