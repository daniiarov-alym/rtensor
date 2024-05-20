use std::io;

pub struct Reader {}

impl Reader {
    pub fn new() -> Self {
        Self {}
    }

    pub fn read_input(&mut self) -> Result<String, &'static str> {
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                Ok(String::new())
            }
            Ok(_) => {
                let trimmed_input = input.trim().to_string();
                match self.validate(&trimmed_input) {
                    Ok(()) => Ok(trimmed_input),
                    Err(err_msg) => Err(err_msg),
                }
            }
            Err(_) => Err("Failed to read input"),
        }
    }

    fn validate(&mut self, input: &str) -> Result<(), &'static str> {
        if input.trim().is_empty() {
            Err("Input cannot be empty")
        } else {
            Ok(())
        }
    }
}
