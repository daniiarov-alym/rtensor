mod core;
mod io;
mod lang;
mod numerical;
mod symbolic;

use io::reader;
use io::writer;
use std::env;

static HELP_MESSAGE: &str = "RTensor -- Computer Algebra System for Tensor Algebra

Usage: rtensor [options..]
\t-f, --filename <filename>\tfilename containing input queries to the system
\t--no-prompt\t\t\tdisable displaying prompt to user
\t-h, --help\t\t\tdisplay this message
\t-o, --output <filename>\t\ttwrite output of the queries to the filename given
\t--verbose\t\t\tturn on verbose printing";

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let mut verbose = false;
    let mut file_input = false;
    let mut input_filename = String::new();
    
    let mut file_output = false;
    let mut output_filename = String::new();
    let mut write_prompt = true;
    
    let mut args_idx = 1 as usize;
    while args_idx < args.len() {
        if args[args_idx] == "--verbose" || args[args_idx] == "-v" {
            verbose = true;
        } else if args[args_idx] == "--filename" || args[args_idx] == "-f" {
            write_prompt = false;
            file_input = true;
            args_idx += 1;
            if args_idx >= args.len() {
                eprintln!("Error: no filename given");
                eprintln!("{}", HELP_MESSAGE);
                return;
            }
            input_filename = args[args_idx].clone();
        } else if args[args_idx] == "--no-prompt" {
            write_prompt = false;            
        } else if args[args_idx] == "-h" || args[args_idx] == "--help" {
            println!("{}", HELP_MESSAGE);
            return;
        } else if args[args_idx] == "--output" || args[args_idx] == "-o" {
            file_output = true;
            args_idx += 1;
            if args_idx >= args.len() {
                eprintln!("Error: no filename given");
                eprintln!("{}", HELP_MESSAGE);
                return;
            }
            output_filename = args[args_idx].clone();
        } else {
            eprintln!("unrecognized option: {}", args[args_idx]);
            eprintln!("{}", HELP_MESSAGE);
            return;
        }
        args_idx += 1;
    }
    
    if verbose && file_output {
        println!("verbose mode does not work with file output, ignoring verbose");
        verbose = false;
    }
    
    let mut reader = reader::Reader::new();
    if file_input {
        reader = reader::Reader::new_with_file(input_filename).unwrap();
    }
    let mut writer = writer::Writer::new();
    if file_output {
        writer = writer::Writer::new_with_file(output_filename).unwrap();
    }
    let mut dispatcher = io::dispatcher::Dispatcher::new();
    loop {
        if write_prompt {
            writer.prompt();
        }
        let read = reader.read_input();
        match read {
            Ok(read) => {
                if read.is_empty() {
                    break;
                }
                let mut tokenizer = lang::tokenizer::Tokenizer::new(&read);
                let tokens = tokenizer.tokenize();
                if let Err(e) = tokens {
                    let _ =  writer.write_output(&format!("Error: {}", e));
                    continue;
                }
                let tokens = tokens.unwrap();
                if verbose {
                    let _ = writer.write_output(&format!("Tokens: {:?}", tokens));
                }
                let mut parser = lang::parser::Parser::new(tokens);
                match parser.parse_impl() {
                    Ok(expr) => {
                        if expr.is_none() {
                            continue;
                        }
                        let expr = expr.unwrap();
                        if verbose {
                            let _ = writer.write_output(&format!("AST: {:?}", expr));
                        }
                        let result = dispatcher.process_expr(expr, verbose);
                        if let Err(e) = &result {
                            let _ = writer.write_output(&format!("Error: {}", e));
                            continue;
                        }
                        if verbose {
                            let _ = writer.write_output(&format!("{:?}", result));
                        }
                        let result = result.unwrap();
                        let _ = writer.write_return_result(result);
                    }
                    Err(err) => {let _ = writer.write_output(&format!("Error: {}", err));}
                }
            }
            Err(e) => {
                writer.write_output(&e).unwrap();
            }
        }
    }
}
