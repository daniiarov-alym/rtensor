use std::io;
use std::io::Write;

use crate::symbolic::SymbolicExpr;

use super::dispatcher::ReturnResult;
pub struct Writer {}

impl Writer {
    pub fn new() -> Self {
        Writer {}
    }

    pub fn prompt(&mut self) {
        print!("> ");
        io::stdout().flush().unwrap();
    }

    pub fn write_output(&mut self, output: &str) -> Result<(), io::Error> {
        println!("{}", output);
        Ok(())
    }

    pub fn write_return_result(&mut self, r: ReturnResult) -> Result<(), io::Error> {
        match r {
            ReturnResult::Nothing => return Ok(()),
            ReturnResult::Tensor(t) => {
                println!("{}", t);
            }
            ReturnResult::Literal(f) => {
                println!("{}", f);
            }
            ReturnResult::TensorShape(s) => {
                println!("{:?}", s);
            }
            ReturnResult::Symbolic(expr) => {
                self.format_symbolic(expr);
            }
        }
        Ok(())
    }

    fn format_symbolic(&mut self, expr: SymbolicExpr) {
        let mut res = String::new();
        self.format_symbolic_impl(&expr, &mut res);
        println!("{}", res)
    }

    fn format_symbolic_impl(&mut self, expr: &SymbolicExpr, out: &mut String) {
        match expr {
            SymbolicExpr::Symbol(s) => {
                out.push_str(&s);
            }
            SymbolicExpr::FunctionCall { name, args } => {
                out.push_str(&name);
                out.push_str("(");
                for i in 0..args.len() {
                    if i > 0 {
                        out.push_str(", ")
                    }
                    self.format_symbolic_impl(&args[i], out);
                }
                out.push_str(")");
            }
            SymbolicExpr::BinaryOp { op, left, right } => {
                self.format_symbolic_impl(left, out);
                out.push_str(&op);
                self.format_symbolic_impl(right, out);
            }
            SymbolicExpr::UnnamedTensor { tensor } => out.push_str(&format!("{}", tensor)),
            SymbolicExpr::UnnamedLiteral { literal } => out.push_str(&format!("{}", literal)),
        }
    }
}
