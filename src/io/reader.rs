use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::path::Path;
pub struct Reader {
    lines: Option<Lines<BufReader<File>>>
}

impl Reader {
    pub fn new() -> Self {
        Self {lines: None}
    }
    
    pub fn new_with_file(filename: String) -> Result<Self, Box<dyn Error>> {
        let lines = read_lines(filename)?;
        Ok(Self{
            lines: Some(lines)
        })
    }

    fn read_input_file(&mut self) -> Result<String, String> {
        let next_line = self.lines.as_mut().unwrap().next();
        if next_line.is_none() {
            return Ok(String::new());
        }
        let next_line = next_line.unwrap();
        match next_line {
            Ok(s) => {
                if (s.is_empty()) {
                    return Ok(" ".to_string());
                }
                return Ok(s)
            }
            Err(e) => {
                return Err(format!("{}",e));
            }
        }
    }
    
    pub fn read_input(&mut self) -> Result<String, String> {
        
        if self.lines.is_some() {
            return self.read_input_file();
        }
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => Ok(String::new()),
            Ok(_) => {
                let trimmed_input = input.trim().to_string();
                if trimmed_input.is_empty() {
                    return Ok(" ".to_string());
                }
                return Ok(trimmed_input);
            }
            Err(_) => Err("Failed to read input".to_string()),
        }
    }

    fn validate(&mut self, input: &str) -> Result<(), &'static str> {
        if input.trim().is_empty() {
            Ok(())
        } else {
            Ok(())
        }
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
