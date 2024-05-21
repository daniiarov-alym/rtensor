use super::ast::Expr;
use super::tokenizer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    current_position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            current_position: 0,
        }
    }

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.current_position)
    }

    fn advance(&mut self) {
        self.current_position += 1;
    }

    fn back(&mut self) {
        self.current_position -= 1;
    }

    // TODO: fix the issue that A=1+#foobar or A=#foobar is valid
    pub fn parse_impl(&mut self) -> Result<Expr, String> {
        if self.tokens.len() == 0 {
            return Err(String::from("empty expression"));
        }
        if self.tokens.len() == 1 {
            // this should be just identifier or literal
            match &self.tokens[0] {
                Token::Identifier(id) => return Ok(Expr::Identifier(id.clone())),
                Token::Literal(f) => return Ok(Expr::Literal(*f)),
                _ => return Err(format!("unexpected token: {:?}", self.tokens[0])),
            }
        }
        let mut expr = Expr::Literal(0.0);
        // otherwise it should be statement or function call
        match &self.tokens[0] {
            Token::Identifier(s) => {
                if self.tokens[1] == Token::Operator("=".to_string()) {
                    if self.tokens.len() - 2 <= self.current_position {
                        return Err("55 malformed statement".to_string());
                    }
                    expr = Expr::Assignment {
                        name: s.clone(),
                        expr: Box::new(Expr::Literal(0.0)),
                    }; // it will be mutated later
                    self.advance();
                    self.advance(); //just workaround so that self.current_token points to value after =
                    match &mut expr {
                        Expr::Assignment { expr, .. } => {
                            if let Err(e) = self.parse_expression_impl_base(expr) {
                                return Err(e);
                            }
                        }
                        _ => {
                            // this branch actually does not happen
                        }
                    }
                // then there is expression
                } else {
                    if let Err(e) = self.parse_expression_impl_base(&mut expr) {
                        return Err(e);
                    }
                }
            }
            Token::Literal(_) => {
                if self.tokens[1] == Token::Operator("=".to_string()) {
                    return Err(format!("unexpected assignment to literal"));
                }
                if let Err(e) = self.parse_expression_impl_base(&mut expr) {
                    return Err(e);
                }
            }
            _ => {
                if let Err(e) = self.parse_expression_impl_base(&mut expr) {
                    return Err(e);
                }
            }
        }

        Ok(expr) // placeholder
    }

    fn parse_expression_impl_base(&mut self, expr: &mut Expr) -> Result<(), String> {
        match self.current_token() {
            Some(Token::Punctuation('[')) => {
                // it is tensor we should first parse it
                // something like self.parse_tensor_impl(expr)
                // and check that there is some to read
                *expr = Expr::Tensor(Vec::new());
                match expr {
                    Expr::Tensor(v) => {
                        if let Err(e) = self.parse_tensor_impl(v) {
                            return Err(e);
                        }
                    }
                    _ => {
                        // this branch actually does not happen
                    }
                }
                self.back();
                if self.tokens.len() - 1 <= self.current_position {
                    // we succesfully have read the whole tensor
                    return Ok(());
                }
                if self.tokens.len() - 2 <= self.current_position {
                    return Err("118 malformed statement".to_string());
                }
                match &self.tokens[self.current_position + 1] {
                    Token::Identifier(s) => return Err(format!("unexpected token {:?}", s)),
                    Token::Operator(s) => {
                        if s == "=" {
                            return Err(format!("unexpected operator ="));
                        }
                        *expr = Expr::BinaryOp {
                            op: s.clone(),
                            left: Box::new(expr.clone()),
                            right: Box::new(Expr::Literal(0.0)),
                        }; // here left should enclose tensor
                        self.advance();
                        self.advance();
                        match expr {
                            Expr::BinaryOp { right, .. } => {
                                if let Err(e) = self.parse_expression_impl_base(right) {
                                    return Err(e);
                                }
                            }
                            _ => {
                                // this branch actually does not happen
                            }
                        }
                    }
                    Token::Literal(f) => return Err(format!("unexpected token {:?}", f)),
                    Token::Punctuation(s) => return Err(format!("unexcpected token: {:?}", s)),
                    Token::Comment(_) => return Ok(()),
                    _ => {
                        return Err(format!(
                            "unexpected token {:?}",
                            self.tokens[self.current_position + 1]
                        ))
                    }
                }
            }
            Some(Token::Identifier(id)) => {
                // if there is -- it is probably an some operation
                // or function call and we should process further
                // something like self.parse_expression_impl_base
                if self.tokens.len() - 1 <= self.current_position {
                    *expr = Expr::Identifier(id.clone());
                    return Ok(());
                }

                if self.tokens.len() - 2 <= self.current_position {
                    return Err("167 malformed statement".to_string());
                }
                match &self.tokens[self.current_position + 1] {
                    Token::Identifier(s) => return Err(format!("unexpected token {:?}", s)),
                    Token::Operator(s) => {
                        if s == "=" {
                            return Err(format!("unexpected operator ="));
                        }
                        *expr = Expr::BinaryOp {
                            op: s.clone(),
                            left: Box::new(Expr::Identifier(id.clone())),
                            right: Box::new(Expr::Literal(0.0)),
                        };
                        self.advance();
                        self.advance();
                        match expr {
                            Expr::BinaryOp { right, .. } => {
                                if let Err(e) = self.parse_expression_impl_base(right) {
                                    return Err(e);
                                }
                            }
                            _ => {
                                // this branch actually does not happen
                            }
                        }
                    }
                    Token::Literal(f) => return Err(format!("unexpected token {:?}", f)),
                    Token::Punctuation(s) => {
                        if s != &'(' {
                            return Err(format!("unexpected token {:?}", s));
                        }
                        // we somehow need to parse function call
                        // ideally also index operator
                        let mut fn_call = Expr::FunctionCall {
                            name: id.clone(),
                            args: Vec::new(),
                        };
                        self.advance();
                        // implement parsing
                        match &mut fn_call {
                            Expr::FunctionCall { args, .. } => {
                                if let Err(e) = self.parse_function_call_args_impls(args) {
                                    return Err(e);
                                }
                            }
                            _ => {
                                // this branch actually does not happen
                            }
                        }
                        if self.tokens.len() - 1 <= self.current_position {
                            *expr = fn_call;
                            return Ok(());
                        }
                        if self.tokens.len() - 2 <= self.current_position {
                            return Err("218 malformed statement".to_string());
                        }
                        match &self.tokens[self.current_position + 1] {
                            Token::Identifier(s) => {
                                return Err(format!("unexpected token {:?}", s))
                            }
                            Token::Operator(s) => {
                                if s == "=" {
                                    return Err(format!("unexpected operator ="));
                                }
                                *expr = Expr::BinaryOp {
                                    op: s.clone(),
                                    left: Box::new(fn_call),
                                    right: Box::new(Expr::Literal(0.0)),
                                }; // here left should enclose tensor
                                self.advance();
                                self.advance();
                                match expr {
                                    Expr::BinaryOp { right, .. } => {
                                        if let Err(e) = self.parse_expression_impl_base(right) {
                                            return Err(e);
                                        }
                                    }
                                    _ => {
                                        // this branch actually does not happen
                                    }
                                }
                            }
                            Token::Literal(f) => return Err(format!("unexpected token {:?}", f)),
                            Token::Punctuation(s) => {
                                return Err(format!("unexcpected token: {:?}", s))
                            }
                            Token::Comment(_) => return Ok(()),
                            _ => {
                                return Err(format!(
                                    "unexpected token {:?}",
                                    self.tokens[self.current_position + 1]
                                ))
                            }
                        }
                    }
                    Token::Comment(_) => return Ok(()),
                    _ => {
                        return Err(format!(
                            "unexpected token {:?}",
                            self.tokens[self.current_position + 1]
                        ))
                    }
                }
            }
            Some(Token::Literal(f_ext)) => {
                // if there is -- it is probably an some operation
                // or function call and we should process further
                // something like self.parse_expression_impl_base
                if self.tokens.len() - 1 <= self.current_position {
                    // there is nothing more
                    *expr = Expr::Literal(*f_ext);
                    return Ok(());
                }
                if self.tokens.len() - 2 <= self.current_position {
                    return Err("278 malformed statement".to_string());
                }

                match &self.tokens[self.current_position + 1] {
                    Token::Identifier(s) => return Err(format!("unexpected token {:?}", s)),
                    Token::Operator(s) => {
                        if s == "=" {
                            return Err(format!("unexpected operator ="));
                        }
                        *expr = Expr::BinaryOp {
                            op: s.clone(),
                            left: Box::new(Expr::Literal(*f_ext)),
                            right: Box::new(Expr::Literal(0.0)),
                        };
                        self.advance();
                        self.advance();
                        match expr {
                            Expr::BinaryOp { right, .. } => {
                                if let Err(e) = self.parse_expression_impl_base(right) {
                                    return Err(e);
                                }
                            }
                            _ => {
                                // this branch actually does not happen
                            }
                        }
                    }
                    Token::Literal(f) => return Err(format!("unexpected token {:?}", f)),
                    Token::Punctuation(s) => return Err(format!("unexpected token {:?}", s)),
                    Token::Comment(_) => return Ok(()),
                    _ => {
                        return Err(format!(
                            "unexpected token {:?}",
                            self.tokens[self.current_position + 1]
                        ))
                    }
                }
            }
            Some(Token::Comment(_)) => {
                return Ok(()); // we ignore comments
            }
            Some(t) => return Err(format!("unexpected token: {:?}", t)),
            None => {
                return Ok(()); // we finished parsing
            }
        }
        return Ok(());
    }

    fn parse_tensor_impl(&mut self, entries: &mut Vec<Expr>) -> Result<(), String> {
        // well defined tensor will have form of nested list
        // Tensor := [Literal, ..., Literal]
        // or
        // Tensor := [Tensor, ..., Tensor]
        self.advance(); // ignore first '['
        let mut expression = Expr::Literal(0.0);
        let mut comma_closed = false;
        let mut paren_closed = false;
        let mut length: usize = 0;
        let mut read_length = false;
        while let Some(token) = self.current_token() {
            match token {
                Token::Literal(f) => {
                    comma_closed = true;
                    expression = Expr::Literal(*f);
                    entries.push(expression);
                    self.advance();
                }
                Token::Punctuation('[') => {
                    comma_closed = true;
                    expression = Expr::Tensor(Vec::new());
                    match &mut expression {
                        Expr::Tensor(v) => {
                            if let Err(e) = self.parse_tensor_impl(v) {
                                return Err(e);
                            }
                            if !read_length {
                                length = v.len();
                            }
                            read_length = true;
                            if (v.len() != length) {
                                return Err(format!(
                                    "invalid length of tensor dimension entry, should be {}, is {}",
                                    length,
                                    v.len()
                                ));
                            }
                        }
                        _ => {
                            // this branch actually does not happen
                        }
                    }
                    entries.push(expression);
                }

                Token::Punctuation(',') => {
                    if !comma_closed {
                        return Err(format!("Unexpected comma in tensor call arguments"));
                    }
                    comma_closed = false;
                    self.advance();
                }

                Token::Punctuation(']') => {
                    comma_closed = true;
                    paren_closed = true;
                    self.advance();
                    return Ok(());
                }
                _ => {
                    return Err(format!(
                        "unexpected token in tensor expression: {:?}",
                        token
                    ));
                }
            }
        }

        if !paren_closed {
            return Err("parenthesis is not closed".to_string());
        }
        Ok(())
    }

    fn parse_function_call_args_impls(&mut self, args: &mut Vec<Expr>) -> Result<(), String> {
        self.advance();
        let mut paren_closed = false;
        let mut comma_closed = false;
        while let Some(token) = self.current_token() {
            let mut expr = Expr::Literal(0.0);
            match &token {
                Token::Identifier(s) => {
                    comma_closed = true;
                    expr = Expr::Identifier(s.clone());
                    args.push(expr);
                    self.advance()
                    // TODO here we need to check that next token is not (
                    // if it is ( then it might be function call inside function call
                }
                Token::Literal(f) => {
                    comma_closed = true;
                    expr = Expr::Literal(*f);
                    args.push(expr);
                    self.advance()
                }
                Token::Punctuation(',') => {
                    if !comma_closed {
                        return Err(format!("Unexpected comma in function call arguments"));
                    }
                    comma_closed = false;
                    self.advance(); // skip comma, but add check that commas are well formed
                }
                Token::Punctuation(')') => {
                    if !comma_closed && args.len() != 0 {
                        return Err(format!("Unexpected comma in function call arguments"));
                    }
                    paren_closed = true;
                    return Ok(());
                }
                // TODO add branch to parse tensor
                _ => {
                    return Err(format!(
                        "Unexpected token in function call arguments: {:?}",
                        token
                    ))
                }
            }
        }
        if !paren_closed {
            return Err("parenthesis is not closed".to_string());
        }
        Ok(())
    }
}
