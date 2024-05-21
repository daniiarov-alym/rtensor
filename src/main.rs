mod core;
mod io;
mod lang;
mod numerical;
mod symbolic;

use io::reader;
use io::writer;
fn main() {
    let mut reader = reader::Reader::new();
    let mut writer = writer::Writer::new();
    let mut dispatcher = io::dispatcher::Dispatcher::new();
    loop {
        writer.prompt();
        let read = reader.read_input();
        match read {
            Ok(read) => {
                if read.is_empty() {
                    break;
                }
                let mut tokenizer = lang::tokenizer::Tokenizer::new(&read);
                let tokens = tokenizer.tokenize();
                //println!("{:?}", tokens);
                let mut parser = lang::parser::Parser::new(tokens);
                match parser.parse_impl() {
                    Ok(expr) => {
                        //println!("Input represents a valid statement: {:?}", expr);
                        let result = dispatcher.process_expr(expr);
                        if let Err(e) = &result {
                            println!("Error: {}", e);
                            continue;
                        }
                        let result = result.unwrap();
                        let _ = writer.write_return_result(result);
                    }
                    Err(err) => println!("Error: {}", err),
                }
            }
            Err(e) => {
                writer.write_output(&e).unwrap();
            }
        }
    }
}
