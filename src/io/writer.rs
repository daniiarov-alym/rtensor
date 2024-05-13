use std::io;
use std::io::Write;

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
                println!("{:?}", s)
            }
            ReturnResult::Symbol(s) => {
                println!("{:?}", s)
            }
        }
        Ok(())
    }
}
