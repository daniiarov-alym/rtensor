#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Identifier(String),
    Operator(String),
    Punctuation(char),
    Literal(f64),
    Comment(String),
}

// TODO: proper tokenization of function calls? parsing fractions
pub struct Tokenizer<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, position: 0 }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.input.chars().nth(self.position) {
            if !c.is_whitespace() {
                break;
            }
            self.position += 1;
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut identifier = String::new();
        while let Some(c) = self.input.chars().nth(self.position) {
            if c.is_alphanumeric() || c == '_' {
                identifier.push(c);
                self.position += 1;
            } else {
                break;
            }
        }
        identifier
    }

    fn read_number(&mut self) -> f64 {
        let mut number_str = String::new();
        while let Some(c) = self.input.chars().nth(self.position) {
            if c.is_ascii_digit() || c == '.' {
                number_str.push(c);
                self.position += 1;
            } else {
                break;
            }
        }
        number_str.parse().unwrap_or(0.0) // Return 0.0 if parsing fails
    }

    fn read_operator(&mut self) -> String {
        let mut operator = String::new();
        while let Some(c) = self.input.chars().nth(self.position) {
            if "+-*/=".contains(c) {
                operator.push(c);
                self.position += 1;
                break;
            } else {
                break;
            }
        }
        operator
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while self.position < self.input.len() {
            self.skip_whitespace();
            if let Some(c) = self.input.chars().nth(self.position) {
                match c {
                    '(' | ')' | '[' | ']' | ',' => {
                        tokens.push(Token::Punctuation(c));
                        self.position += 1;
                    }
                    'a'..='z' | 'A'..='Z' | '_' => {
                        let identifier = self.read_identifier();
                        // TODO: add check that identifier is keyword
                        tokens.push(Token::Identifier(identifier));
                    }
                    '0'..='9' => {
                        let number = self.read_number();
                        tokens.push(Token::Literal(number));
                    }
                    '+' | '-' | '*' | '/' | '=' => {
                        // it would be unary - if A=-1 
                        if c == '-' && self.position == 0 && self.input.len() > 1 {
                            self.position += 1;
                            match self.input.chars().nth(self.position).unwrap() {
                                '0'..='9' => {
                                    let number = self.read_number();
                                    tokens.push(Token::Literal(-number));
                                }
                                _ => {
                                    continue;
                                }
                            }
                            continue;
                        }
                        if c == '-' && self.position > 0 && self.position < self.input.len()-1 {
                            match self.input.chars().nth(self.position-1).unwrap() {
                                '(' | '[' | ',' => {
                                    
                                    self.position += 1;
                                    match self.input.chars().nth(self.position).unwrap() {
                                        '0'..='9' => {
                                            let number = self.read_number();
                                            tokens.push(Token::Literal(-number));
                                            continue;
                                        }
                                        _ => {
                                            continue;
                                        }
                                    }
                                }
                                _ => {
                                    
                                }
                            }
                        } 
                        let operator = self.read_operator();
                        tokens.push(Token::Operator(operator));
                    }
                    '#' => {
                        let mut comment = String::new();
                        while let Some(next_char) = self.input.chars().nth(self.position) {
                            if next_char == '\n' {
                                break;
                            }
                            comment.push(next_char);
                            self.position += 1;
                        }
                        tokens.push(Token::Comment(comment));
                    }
                    _ => {
                        self.position += 1; // skip unsupported characters
                    }
                }
            }
        }
        tokens
    }
}
