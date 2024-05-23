use crate::core::datatypes::{Idx, Tensor, TensorBuilder};
use crate::lang::ast::Expr;
use crate::symbolic::SymbolicExpr;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ReturnResult {
    Nothing,
    Literal(f64),
    Tensor(Tensor),
    TensorShape(Vec<usize>),
    Symbolic(SymbolicExpr),
}
// TODO: solve precedence issues

pub struct Dispatcher {
    // dispatcher is also to store variables
    storage: HashMap<String, ReturnResult>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Dispatcher {
            storage: HashMap::new(),
        }
    }

    pub fn process_expr(&mut self, expr: Expr, verbose: bool) -> Result<ReturnResult, String> {
        // this is base start process
        match &expr {
            Expr::Identifier(s) => {
                let val = self.storage.get(s);
                if val.is_none() {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::Symbol(s.clone())));
                }
                return Ok(val.unwrap().clone());
            }
            Expr::Literal(f) => {
                return Ok(ReturnResult::Literal(*f));
            }
            Expr::Tensor(_) => {
                return self.process_tensor_definition(&expr);
            }
            Expr::Assignment { name, expr } => {
                // here logic is more complicated
                // process expression
                // assign its result to name and store it in HashMap
                let r = self.process_expr_inner(expr)?;
                self.storage.insert(name.clone(), r);
            }
            Expr::BinaryOp { .. } => {
                // here logic is:
                // check if operation is supported
                // process both expression and apply operation to their result
                return self.process_binary_op(&expr);
            }
            Expr::FunctionCall { .. } => {
                // check if function is supported
                // process args
                // call the function with args
                return self.process_fn_call(&expr);
            }
        }

        return Ok(ReturnResult::Nothing);
    }

    // this is inner implementation of process_expr for non-base calls
    fn process_expr_inner(&mut self, expr: &Expr) -> Result<ReturnResult, String> {
        match &expr {
            Expr::Identifier(s) => {
                let val = self.storage.get(s);
                if val.is_none() {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::Symbol(s.clone())));
                }
                return Ok(val.unwrap().clone());
            }
            Expr::Literal(f) => {
                return Ok(ReturnResult::Literal(*f));
            }
            Expr::Tensor(_) => return self.process_tensor_definition(&expr),
            Expr::Assignment { .. } => {
                return Err("unexpected assignment in subexpression".to_string());
            }
            Expr::BinaryOp { .. } => {
                // here logic is:
                // check if operation is supported
                // process both expression and apply operation to their result

                //let left_operand = self.process_expr_inner(left)?;
                //let right_operand = self.process_expr_inner(right)?;

                // now assume we somehow looked up and found out that operation is supported
                // return self.process_op(op, left_operand, right_operand)
                return self.process_binary_op(expr);
            }
            Expr::FunctionCall { .. } => {
                // check if function is supported
                // process args
                // call the function with args

                //let mut evaluated: Vec<ReturnResult> = Vec::new();
                //for expr in args.iter() {
                //	let ev_expr = self.process_expr_inner(expr)?;
                //	evaluated.push(ev_expr);
                //}
                // and do function call
                // something like self.process_fn_call(fn, evaluated)
                return self.process_fn_call(expr);
            }
        }
    }

    // function to process tensor definition
    fn process_tensor_definition(&self, expr: &Expr) -> Result<ReturnResult, String> {
        let mut builder = TensorBuilder::new();
        if let Err(e) = self.process_tensor_definition_impl(expr, &mut builder, true) {
            return Err(e);
        }
        Ok(ReturnResult::Tensor(builder.build()))
    }

    fn process_tensor_definition_impl(
        &self,
        expr: &Expr,
        builder: &mut TensorBuilder,
        mut first: bool,
    ) -> Result<(), String> {
        match expr {
            Expr::Tensor(v) => {
                for i in 0..v.len() {
                    if first {
                        builder.shape.push(v.len())
                    }
                    if let Err(e) = self.process_tensor_definition_impl(&v[i], builder, first) {
                        return Err(e);
                    }
                    first = false;
                }
                return Ok(());
            }
            Expr::Literal(f) => {
                // if we are in this branch we should be on the lowest level of parsing
                builder.raw_data.push(*f);
                return Ok(());
            }
            _ => {
                return Err("unexpected expression type".to_string());
            }
        }
    }

    fn process_fn_call(&mut self, expr: &Expr) -> Result<ReturnResult, String> {
        match expr {
            Expr::FunctionCall { name, args } => {
                // we support following functions
                // outer(A, B) -> Tensor
                // inner(A, B) -> Float
                // transpose(A) -> Tensor
                // index(A, indices...) -> Tensor|Float (?)
                // rank(A) -> unsigned
                // shape(A) -> (unsigned,...)
                // other...
                let mut evaluated: Vec<ReturnResult> = Vec::new();
                for expr in args.iter() {
                    let ev_expr = self.process_expr_inner(expr)?;
                    evaluated.push(ev_expr);
                }
                if name == "rank" {
                    return self.process_rank_call(evaluated);
                } else if name == "shape" {
                    return self.process_shape_call(evaluated);
                } else if name == "index" {
                    return self.process_index_call(evaluated);
                } else if name == "outer" {
                    return self.process_outer_call(evaluated);
                } else if name == "inner" {
                    return self.process_inner_call(evaluated);
                } else if name == "transpose" {
                    return self.process_transpose_call(evaluated);
                } else if name == "hosvd" {
                    return self.process_hosvd(evaluated);
                } else if name == "evaluate" {
                    return self.evaluate_symbolic(evaluated);
                } else if name == "solve" {
                    return self.process_solve(evaluated);
                } else {
                    return Err(format!("function {:?} is not recognized", name));
                }
                
            }
            _ => {
                return Err("unexpected expression type".to_string());
            }
        }
    }

    fn process_binary_op(&mut self, expr: &Expr) -> Result<ReturnResult, String> {
        match expr {
            Expr::BinaryOp { op, left, right } => {
                // we support following binary ops
                // Literal * Tensor
                // Tensor + Tensor
                // Tensor - Tensor
                // Literal + Literal
                // Literal - Literal
                // Literal * Literal
                // Literal / Literal
                let left_eval = self.process_expr_inner(left)?;
                let right_eval = self.process_expr_inner(right)?;
                if op == "+" {
                    return self.process_addition(left_eval, right_eval);
                } else if op == "-" {
                    return self.process_subtraction(left_eval, right_eval);
                } else if op == "*" {
                    return self.process_multiplication(left_eval, right_eval);
                } else if op == "/" {
                    return self.process_division(left_eval, right_eval);
                } else {
                    return Err(format!("operation {:?} is not recognized", op));
                }
            }
            _ => {
                return Err("unexpected expression type".to_string());
            }
        }
    }

    fn process_addition(
        &self,
        left: ReturnResult,
        right: ReturnResult,
    ) -> Result<ReturnResult, String> {
        // if left is scalar right should be scalar
        match left {
            ReturnResult::Literal(lhs) => match right {
                ReturnResult::Literal(rhs) => return Ok(ReturnResult::Literal(lhs + rhs)),
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "+".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedLiteral { literal: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            ReturnResult::Tensor(lhs) => match right {
                ReturnResult::Tensor(rhs) => return Ok(ReturnResult::Tensor(lhs + rhs)),
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "+".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedTensor { tensor: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => return Err("unexpected operand type".to_string()),
            },
            ReturnResult::Symbolic(lhs) => match right {
                ReturnResult::Tensor(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "+".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedTensor { tensor: rhs }),
                    }));
                }
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "+".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedLiteral { literal: rhs }),
                    }));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "+".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }));
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            _ => return Err("unexpected operand type".to_string()),
        }
    }

    fn process_subtraction(
        &self,
        left: ReturnResult,
        right: ReturnResult,
    ) -> Result<ReturnResult, String> {
        // if left is scalar right should be scalar
        match left {
            ReturnResult::Literal(lhs) => match right {
                ReturnResult::Literal(rhs) => return Ok(ReturnResult::Literal(lhs - rhs)),
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "-".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedLiteral { literal: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            ReturnResult::Tensor(lhs) => match right {
                ReturnResult::Tensor(rhs) => return Ok(ReturnResult::Tensor(lhs - rhs)),
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "-".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedTensor { tensor: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => return Err("unexpected operand type".to_string()),
            },
            ReturnResult::Symbolic(lhs) => match right {
                ReturnResult::Tensor(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "-".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedTensor { tensor: rhs }),
                    }));
                }
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "-".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedLiteral { literal: rhs }),
                    }));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "-".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }));
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            _ => return Err("unexpected operand type".to_string()),
        }
    }

    fn process_multiplication(
        &self,
        left: ReturnResult,
        right: ReturnResult,
    ) -> Result<ReturnResult, String> {
        match left {
            ReturnResult::Literal(lhs) => match right {
                ReturnResult::Literal(rhs) => return Ok(ReturnResult::Literal(lhs * rhs)),
                ReturnResult::Tensor(rhs) => {
                    return Ok(ReturnResult::Tensor(rhs.scalar_multiply(lhs)));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "*".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedLiteral { literal: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => return Err("unexpected operand type".to_string()),
            },
            ReturnResult::Tensor(lhs) => match right {
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Tensor(lhs.scalar_multiply(rhs)));
                }
                ReturnResult::Tensor(rhs) => {
                    return Ok(ReturnResult::Tensor(lhs.outer_product(&rhs)))
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "*".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedTensor { tensor: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => return Err("unexpected operand type".to_string()),
            },
            ReturnResult::Symbolic(lhs) => match right {
                ReturnResult::Tensor(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "*".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedTensor { tensor: rhs }),
                    }));
                }
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "*".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedLiteral { literal: rhs }),
                    }));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "*".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }));
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            _ => {}
        }
        Ok(ReturnResult::Nothing)
    }

    fn process_division(
        &self,
        left: ReturnResult,
        right: ReturnResult,
    ) -> Result<ReturnResult, String> {
        // if left is scalar right should be scalar
        match left {
            ReturnResult::Literal(lhs) => match right {
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Literal(lhs / rhs));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "/".to_string(),
                        left: Box::new(SymbolicExpr::UnnamedLiteral { literal: lhs }),
                        right: Box::new(rhs),
                    }))
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },

            ReturnResult::Symbolic(lhs) => match right {
                ReturnResult::Tensor(_) => {
                    return Err("Tensor division is not supported".to_string());
                }
                ReturnResult::Literal(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "/".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(SymbolicExpr::UnnamedLiteral { literal: rhs }),
                    }));
                }
                ReturnResult::Symbolic(rhs) => {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::BinaryOp {
                        op: "/".to_string(),
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    }));
                }
                _ => {
                    return Err("unexpected operand type".to_string());
                }
            },
            _ => return Err("Tensor division is not supported".to_string()),
        }
    }

    fn process_rank_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 1 {
            return Err("rank accepts only 1 argument".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => {
                return Ok(ReturnResult::Literal(t.rank() as f64));
            }
            ReturnResult::Literal(_) => {
                return Ok(ReturnResult::Literal(0.0));
            }
            ReturnResult::Symbolic(t) => {
                let mut ev_args = Vec::<SymbolicExpr>::new();
                ev_args.push(t.clone());
                let res = SymbolicExpr::FunctionCall {
                    name: "rank".to_string(),
                    args: ev_args,
                };
                return Ok(ReturnResult::Symbolic(res));
            }
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
    }

    fn process_shape_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        // TODO: implement
        if args.len() != 1 {
            return Err("invalid arguments".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => {
                return Ok(ReturnResult::TensorShape(t.shape()));
            }
            ReturnResult::Symbolic(t) => {
                let mut ev_args = Vec::<SymbolicExpr>::new();
                ev_args.push(t.clone());
                let res = SymbolicExpr::FunctionCall {
                    name: "shape".to_string(),
                    args: ev_args,
                };
                return Ok(ReturnResult::Symbolic(res));
            }
            ReturnResult::Literal(_) => {
                return Ok(ReturnResult::TensorShape(Vec::new()));
            }
            _ => {
                return Err(format!("argument {:?} does not have shape", args[0]));
            }
        }
    }

    fn process_index_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() < 1 {
            return Err("no arguments provided".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => {
                let mut index = Vec::<usize>::new();
                for i in 1..args.len() {
                    match args[i] {
                        ReturnResult::Literal(f) => {
                            index.push(f as usize);
                        }
                        _ => return Err(format!("invalid argument: {:?}", args[i])),
                    }
                }
                let idx = Idx::new(&index);
                return Ok(ReturnResult::Literal(t[idx]));
            }
            ReturnResult::Symbolic(t) => {
                let mut res = Vec::<SymbolicExpr>::new();
                res.push(t.clone());
                for i in 1..args.len() {
                    match args[i] {
                        ReturnResult::Literal(f) => {
                            res.push(SymbolicExpr::UnnamedLiteral { literal: f })
                        }

                        _ => return Err(format!("invalid argument: {:?}", args[i])),
                    }
                }
                return Ok(ReturnResult::Symbolic(SymbolicExpr::FunctionCall {
                    name: "index".to_string(),
                    args: res,
                }));
            }
            _ => {
                return Err(format!("argument should be a tensor"));
            }
        }
        // TODO: implement
    }

    fn process_outer_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 2 {
            return Err("outer product accepts only 2 arguments".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => match &args[1] {
                ReturnResult::Tensor(p) => {
                    return Ok(ReturnResult::Tensor(t.outer_product(p)));
                }
                ReturnResult::Symbolic(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: t.clone() });
                    ev_args.push(p.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "outer".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            },
            ReturnResult::Symbolic(t) => match &args[1] {
                ReturnResult::Tensor(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: p.clone() });
                    let res = SymbolicExpr::FunctionCall {
                        name: "outer".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                ReturnResult::Symbolic(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(p.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "outer".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            },
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
    }

    fn process_inner_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 2 {
            return Err("inner product accepts only 2 arguments".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => match &args[1] {
                ReturnResult::Tensor(p) => {
                    return Ok(ReturnResult::Literal(t.inner_product(p)));
                }
                ReturnResult::Symbolic(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: t.clone() });
                    ev_args.push(p.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "inner".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            },

            ReturnResult::Symbolic(t) => match &args[1] {
                ReturnResult::Tensor(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: p.clone() });
                    let res = SymbolicExpr::FunctionCall {
                        name: "inner".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                ReturnResult::Symbolic(p) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(p.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "inner".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            },
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
    }

    fn process_transpose_call(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 1 {
            return Err("transpose accepts only 1 argument".to_string());
        }
        match &args[0] {
            ReturnResult::Literal(_) => return Ok(args[0].clone()),
            ReturnResult::Tensor(t) => {
                return Ok(ReturnResult::Tensor(t.tensor_transpose()));
            }
            ReturnResult::Symbolic(t) => {
                let mut ev_args = Vec::<SymbolicExpr>::new();
                ev_args.push(t.clone());
                let res = SymbolicExpr::FunctionCall {
                    name: "transpose".to_string(),
                    args: ev_args,
                };
                return Ok(ReturnResult::Symbolic(res));
            }
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
    }

    fn process_hosvd(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 1 {
            return Err("hosvd accepts only 1 argument".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => {
                t.hosvd();
                return Ok(ReturnResult::Nothing);
            }

            ReturnResult::Symbolic(t) => {
                let mut ev_args = Vec::<SymbolicExpr>::new();
                ev_args.push(t.clone());
                let res = SymbolicExpr::FunctionCall {
                    name: "hosvd".to_string(),
                    args: ev_args,
                };
                return Ok(ReturnResult::Symbolic(res));
            }
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
    }
    
    fn process_solve(&self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() != 2 {
            return Err("solve accepts exactly two arguments".to_string());
        }
        match &args[0] {
            ReturnResult::Tensor(t) => match &args[1] {
                ReturnResult::Tensor(b) => {
                    return Ok(ReturnResult::Tensor(t.solve_matrix_vector_equation(b)?));
                }
                ReturnResult::Symbolic(b) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: t.clone()});
                    ev_args.push(b.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "solve".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            }
            ReturnResult::Symbolic(t) =>  match &args[1] {
                ReturnResult::Tensor(b) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(SymbolicExpr::UnnamedTensor { tensor: b.clone()});
                    let res = SymbolicExpr::FunctionCall {
                        name: "solve".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                ReturnResult::Symbolic(b) => {
                    let mut ev_args = Vec::<SymbolicExpr>::new();
                    ev_args.push(t.clone());
                    ev_args.push(b.clone());
                    let res = SymbolicExpr::FunctionCall {
                        name: "solve".to_string(),
                        args: ev_args,
                    };
                    return Ok(ReturnResult::Symbolic(res));
                }
                _ => {
                    return Err(format!("invalid argument: {:?}", args[0]));
                }
            }
            _ => {
                return Err(format!("invalid argument: {:?}", args[0]));
            }
        }
        
        Ok(ReturnResult::Nothing)
    }

    // TODO: add function to evaluate symbols
    // the idea is that given arguments (SymbolicExpr,  arg1, ..., argN) we substitute each symbol in SymbolicExpr by arg in order
    // for example evaluate(A+B, a, b) should be the same as a+b
    // 
    fn evaluate_symbolic(&mut self, args: Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if args.len() <= 1 {
            return Err("evaluate expects at least 2 arguments".to_string());
        }
        match &args[0] {
            ReturnResult::Symbolic(expr) => {
                let mut args_idx: usize = 1;
                match expr {
                    SymbolicExpr::Symbol(symbol) => {
                        return Ok(args[1].clone());
                    }
                    
                    SymbolicExpr::BinaryOp { op, left, right } => {
                        // we have to evaluate left and right
                        let left = self.evaluate_symbolic_impl(left, &mut args_idx, &args)?;
                        let right = self.evaluate_symbolic_impl(right, &mut args_idx, &args)?;
                        if op == "+" {
                            return self.process_addition(left, right);
                        } else if op == "-" {
                            return self.process_subtraction(left, right);
                        } else if op == "*" {
                            return self.process_multiplication(left, right);
                        } else if op == "/" {
                            return self.process_division(left, right);
                        } else {
                            return Err(format!("operation {:?} is not recognized", op));
                        }
                    }
                    SymbolicExpr::FunctionCall { name, args: params } => {
                        // we have to evaluate args
                        let mut evaluated = Vec::new();
                        for i in 0..params.len() {
                            evaluated.push(self.evaluate_symbolic_impl(&params[i], &mut args_idx, &args)?);
                        }
                        /*if name == "evaluate" {
                            return Err("recursive call of evaluate detected, aborting".to_string());
                        }*/
                        // call function here
                        if name == "rank" {
                            return self.process_rank_call(evaluated);
                        } else if name == "shape" {
                            return self.process_shape_call(evaluated);
                        } else if name == "index" {
                            return self.process_index_call(evaluated);
                        } else if name == "outer" {
                            return self.process_outer_call(evaluated);
                        } else if name == "inner" {
                            return self.process_inner_call(evaluated);
                        } else if name == "transpose" {
                            return self.process_transpose_call(evaluated);
                        } else if name == "hosvd" {
                            return self.process_hosvd(evaluated);
                        } else if name == "solve" {
                            return self.process_solve(evaluated);
                        } else if name == "evaluate" {
                            return self.process_solve(evaluated);
                        } else {
                            return Err(format!("function {:?} is not recognized", name));
                        }
                    }
                    _ => {
                        return Err(format!("line invalid argument to evaluate: {:?}", expr))
                    }
                }
            }
            _ => {
                return Err(format!("cannot evaluate: {:?}", args[0]));
            }
        }
        
    }
    
    
    fn evaluate_symbolic_impl(&self, expr: &SymbolicExpr, args_idx: &mut usize, args: &Vec<ReturnResult>) -> Result<ReturnResult, String> {
        if *args_idx > args.len() {
            return Err("too many arguments to evaluate".to_string())
        }
        match expr {
            SymbolicExpr::Symbol(s) => {
                
                *args_idx += 1;
                if (*args_idx > args.len()) {
                    return Ok(ReturnResult::Symbolic(SymbolicExpr::Symbol(s.clone())));
                }
                return Ok(args[*args_idx-1].clone());
            }
            SymbolicExpr::BinaryOp { op, left, right } => {
                let left = self.evaluate_symbolic_impl(left, args_idx, &args)?;
                let right = self.evaluate_symbolic_impl(right, args_idx, &args)?;
                if op == "+" {
                    return self.process_addition(left, right);
                } else if op == "-" {
                    return self.process_subtraction(left, right);
                } else if op == "*" {
                    return self.process_multiplication(left, right);
                } else if op == "/" {
                    return self.process_division(left, right);
                } else {
                    return Err(format!("operation {:?} is not recognized", op));
                }
            }
            SymbolicExpr::FunctionCall { name, args: params } => {
                // we have to evaluate args
                let mut evaluated = Vec::new();
                for i in 0..params.len() {
                    evaluated.push(self.evaluate_symbolic_impl(&params[i], args_idx, &args)?);
                }
                /*if name == "evaluate" {
                    return Err("recursive call of evaluate detected, aborting".to_string());
                }*/
                // call function here
                if name == "rank" {
                    return self.process_rank_call(evaluated);
                } else if name == "shape" {
                    return self.process_shape_call(evaluated);
                } else if name == "index" {
                    return self.process_index_call(evaluated);
                } else if name == "outer" {
                    return self.process_outer_call(evaluated);
                } else if name == "inner" {
                    return self.process_inner_call(evaluated);
                } else if name == "transpose" {
                    return self.process_transpose_call(evaluated);
                } else if name == "hosvd" {
                    return self.process_hosvd(evaluated);
                } else if name == "solve" {
                    return self.process_solve(evaluated);
                } else if name == "evaluate" {
                    return self.process_solve(evaluated);
                } else {
                    return Err(format!("function {:?} is not recognized", name));
                }
            }
            
            SymbolicExpr::UnnamedLiteral { literal } => {
                return Ok(ReturnResult::Literal(*literal));
            }
            
            SymbolicExpr::UnnamedTensor { tensor }  => {
                return Ok(ReturnResult::Tensor(tensor.clone()));
            }
        }    
    }
}
