
use std::num::{Int, Float};
use std::collections::HashMap;
use regex::Regex;

#[deriving(Clone, Show, PartialEq, PartialOrd)]
pub enum Val<T> {
	Int(i32),
	Float(f64),
	Str(String),
	Custom(T),
}
impl<T> Val<T> {
	pub fn int(self) -> Option<i32> {
		match self {
			Val::Int(val) => Some(val),
			_             => None,
		}
	}
	pub fn float(self) -> Option<f64> {
		match self {
			Val::Float(val) => Some(val),
			_               => None,
		}
	}
	pub fn str(self) -> Option<String> {
		match self {
			Val::Str(val) => Some(val),
			_             => None,
		}
	} // TODO: macro
}

#[deriving(Clone, Show, Eq, PartialEq, Ord, PartialOrd)]
enum Op {
	Add, Sub, Mul, Div, Pow, Mod,
	And,  Or,  Eq, Neq,  Lt,  Gt, Leq, Geq,
	Paren,
}
impl Op {
	fn get_precidence(&self) -> i32 {
		match *self {
			Op::Paren                           => 0,
			Op::And | Op::Or                    => 1,
			Op::Eq  | Op::Neq                   => 2,
			Op::Leq | Op::Geq | Op::Lt | Op::Gt => 3,
			Op::Add | Op::Sub                   => 4,
			Op::Mul | Op::Div | Op::Mod         => 5,
			Op::Pow                             => 6,
		}
	}
	fn is_right_assoc(&self) -> bool {
		*self == Op::Pow
	}
}

#[deriving(Clone, Show, PartialEq, PartialOrd)]
enum Token<T> {
	Op(Op),
	Var(uint),
	Val(Val<T>),
}

#[deriving(Clone, Show, Eq, PartialEq, Ord, PartialOrd)]
pub enum Error {
	MismatchedParenthesis,
	MisplacedComma,
	UnknownToken(String),
} // TODO: impl Error

#[allow(dead_code)]
// An expr parser. Currently useless (will eventually store custom functions).
pub struct ExprParser<T> {
	functions: Vec<()>,
	function_name_map: HashMap<String, uint>,
}

// Parse string into expression, without having to create an ExprParser.
// This is the main important function probably.
pub fn parse(expression: &str) -> Result<Expr<()>, Error> {
	let parser: ExprParser<()> = ExprParser::new();
	parser.parse(expression)
}
impl<T: Clone> ExprParser<T> {
	pub fn new() -> ExprParser<T> {
		ExprParser { functions: Vec::new(), function_name_map: HashMap::new() }
	}
	/*pub fn add_function(&mut self, name: String, f: |Vec<Val<T>>|:'a -> Option<Val<T>>, num_args: u8) {
		self.function_name_map.insert(name, self.functions.len());
		self.functions.push(f);
	}*/
	pub fn parse(&self, expression: &str) -> Result<Expr<T>, Error> {
		let mut expr: Expr<T> = Expr {
			rpn_stack: Vec::new(),
			variables: Vec::new(),
			variable_name_map: HashMap::new(),
		};
		let mut op_stack: Vec<Op> = Vec::new();

		// Parse each token.
		//let re = regex!(r#"[\d\w\._]+|[!<>=]=|&&|\|\||"[^"]+"|[^\s]"#); // TODO: when fixed
		let mut token = "";
		let re = Regex::new(r#"[\d\w\._]+|[!<>=]=|&&|\|\||"[^"]+"|[^\s]"#).unwrap();
		for (start, end) in re.find_iter(expression) {
			let next_token = expression[start..end];
			if token != "" {
				match self.parse_token(&mut expr, &mut op_stack, token, next_token) {
					Some(err) => return Err(err),
					None => (),
				}
			}
			token = next_token;
		}
		match self.parse_token(&mut expr, &mut op_stack, token, "") {
			Some(err) => return Err(err),
			None => (),
		}

		// Toss the operator stack into the rpn stack.
		loop {
			match op_stack.pop() {
				Some(op) => {
					if op == Op::Paren { return Err(Error::MismatchedParenthesis); }
					expr.rpn_stack.push(Token::Op(op));
				},
				None => break,
			}
		}

		// TODO: optimization

		Ok(expr)
	}
	fn parse_token(&self, expr: &mut Expr<T>,
		           op_stack: &mut Vec<Op>,
		           token: &str, next_token: &str)
	               -> Option<Error> {
		
		if token.is_empty() { return None; }

		let mut chars  = token.chars();
		let first_char = chars.next().unwrap();

		if first_char.is_digit(10) || first_char == '.' {
			// it's assumed to be a number if it starts a digit
			return self.parse_number(expr, token);
		} else if first_char.is_alphabetic() || first_char == '_' {
			// it's assumed to be a variable or function if it starts with a letter or underscore
			if next_token == "(" {
				return self.parse_function(expr, op_stack, token);
			} else {
				return self.parse_variable(expr, token);
			}
		} else if first_char == '"' && token.len() >= 2 && chars.next_back().unwrap() == '"' {
			// it's assumed to be a string if it starts and ends with quotes
			return self.parse_string(expr, token);
		} else {
			// otherwise it's an operator
			return self.parse_operator(expr, op_stack, token);
		}
	}
	fn parse_number(&self, expr: &mut Expr<T>, token: &str) -> Option<Error> {
		if token[].contains(".") {
			// is a float
			match from_str(token) {
				Some(val) => expr.rpn_stack.push(Token::Val(Val::Float(val))),
				None      => return Some(Error::UnknownToken(token.to_string())),
			}
		} else {
			// is an int
			match from_str(token) {
				Some(val) => expr.rpn_stack.push(Token::Val(Val::Int(val))),
				None      => return Some(Error::UnknownToken(token.to_string())),
			}
		}
		None
	}
	fn parse_function(&self, _expr: &mut Expr<T>,
		              _op_stack: &mut Vec<Op>, token: &str)
		              -> Option<Error> {
		/*match self.function_name_map.get(token) {
			Some(i) => op_stack.push(Op::Func(i)),
			None    => return Some(Error::UnknownToken(token.to_string())),
		}*/
		Some(Error::UnknownToken(token.to_string()))
	}
	fn parse_variable(&self, expr: &mut Expr<T>, token: &str) -> Option<Error> {
		// make sure it's all alphanumeric
		for c in token.chars() {
			if !c.is_alphanumeric() {
				return Some(Error::UnknownToken(token.to_string()));
			}
		}

		match expr.variable_name_map.get(token) {
			Some(i) => {
				// variable already exists
				expr.rpn_stack.push(Token::Var((*i)));
				return None;
			},
			None => (),
		};
		let i = expr.variables.len();
		expr.variables.push(None);
		expr.variable_name_map.insert(token.to_string(), i);
		expr.rpn_stack.push(Token::Var((i)));
		None
	}
	fn parse_string(&self, expr: &mut Expr<T>, token: &str) -> Option<Error> {
		expr.rpn_stack.push(Token::Val(Val::Str(token[1..(token.len() - 1)].to_string())));
		None
	}
	fn parse_operator(&self, expr: &mut Expr<T>,
		              op_stack: &mut Vec<Op>, token: &str)
		              -> Option<Error> {
		match token {
			"+"  => self.push_op(expr, op_stack, Op::Add),
			"-"  => self.push_op(expr, op_stack, Op::Sub),
			"*"  => self.push_op(expr, op_stack, Op::Mul),
			"/"  => self.push_op(expr, op_stack, Op::Div),
			"^"  => self.push_op(expr, op_stack, Op::Pow),
			"%"  => self.push_op(expr, op_stack, Op::Mod),
			"<"  => self.push_op(expr, op_stack, Op::Lt),
			">"  => self.push_op(expr, op_stack, Op::Gt),
			"("  => op_stack.push(Op::Paren),
			")"  => return self.pop_paren(expr, op_stack),
			","  => return self.comma(expr, op_stack),
			"==" => self.push_op(expr, op_stack, Op::Eq),
			"!=" => self.push_op(expr, op_stack, Op::Neq),
			"<=" => self.push_op(expr, op_stack, Op::Leq),
			">=" => self.push_op(expr, op_stack, Op::Geq),
			"&&" => self.push_op(expr, op_stack, Op::And),
			"||" => self.push_op(expr, op_stack, Op::Or),
			_ => return Some(Error::UnknownToken(token.to_string())),
		}
		None
	}
	fn push_op(&self, expr: &mut Expr<T>, op_stack: &mut Vec<Op>, op: Op) {
		let precidence = op.get_precidence() * 2 + op.is_right_assoc() as i32;
		loop {
			if op_stack.is_empty() || op_stack[op_stack.len() - 1] == Op::Paren {
				break;
			} else {
				if precidence <= op_stack[op_stack.len() - 1].get_precidence() * 2 {
					expr.rpn_stack.push(Token::Op(op_stack.pop().unwrap()));
				} else { break; }
			}
		}
		op_stack.push(op);
	}
	fn pop_paren(&self, expr: &mut Expr<T>, op_stack: &mut Vec<Op>) -> Option<Error> {
		loop {
			match op_stack.pop() {
				None => return Some(Error::MismatchedParenthesis),
				Some(op) => if op == Op::Paren {
					break
				} else {
					expr.rpn_stack.push(Token::Op(op));
				},
			}
		}
		/*if !op_stack.is_empty() {
			match op_stack[op_stack.len() - 1] {
				Op::Func(i) => expr.rpn_stack.push(Token::Op(Op::Func(i))),
				_ => (),
			}
		}*/
		None
	}
	fn comma(&self, expr: &mut Expr<T>, op_stack: &mut Vec<Op>) -> Option<Error> {
		loop {
			if op_stack.is_empty() {
				return Some(Error::MisplacedComma);
			}
			if op_stack[op_stack.len() - 1] == Op::Paren {
				break;
			}
			expr.rpn_stack.push(Token::Op(op_stack.pop().unwrap()));
		}
		None
	}
}

// An expression.
pub struct Expr<T> {
	rpn_stack: Vec<Token<T>>,
	variables: Vec<Option<Val<T>>>,
	variable_name_map: HashMap<String, uint>,
}
impl<T: Clone> Expr<T> {

	// Sets the value of the given variable to an integer.
	pub fn set_variable_int(&mut self, name: &str, val: i32) {
		match self.variable_name_map.get(name) {
			Some(i) => self.variables[*i] = Some(Val::Int(val)),
			None    => (),
		}
	}

	// Sets the value of the given variable to a floating point value.
	pub fn set_variable_float(&mut self, name: &str, val: f64) {
		match self.variable_name_map.get(name) {
			Some(i) => self.variables[*i] = Some(Val::Float(val)),
			None    => (),
		}
	}

	// Sets the value of the given variable to a string.
	pub fn set_variable_str(&mut self, name: &str, val: String) {
		match self.variable_name_map.get(name) {
			Some(i) => self.variables[*i] = Some(Val::Str(val)),
			None    => (),
		}
	}

	// Sets all the variables in this expression.
	pub fn set_variables(&mut self, f: |&str| -> Val<T>) {
		for (name, idx) in self.variable_name_map.iter() {
			self.variables[*idx] = Some(f(name[]));
		}
	}

	// Evaluates the expression.
	// Will return None if any variables are undefined or any operations are invalid.
	// If it returns a valid value once, it should continue to do so
	// unless you change the type of any of the variables.
	pub fn evaluate(&self) -> Option<Val<T>> {
		let mut eval_stack: Vec<Val<T>> = Vec::new();
		for token in self.rpn_stack.iter() {
			match *token {
				Token::Op(ref op)   => if !Expr::operate(&mut eval_stack, op.clone()) { return None; }, // TODO: what
				Token::Val(ref val) => eval_stack.push(val.clone()),
				Token::Var(i)  => match self.variables[i] {
					None          => return None,
					Some(ref val) => eval_stack.push(val.clone()),
				},
			}
		}
		eval_stack.pop()
	}
	fn operate(eval_stack: &mut Vec<Val<T>>, op: Op) -> bool {
		if eval_stack.len() < 2 { return false; }

		let b = eval_stack.pop().unwrap();
		let a = eval_stack.pop().unwrap();

		let res = match a {
			Val::Int(a_val)   => match b {
				  Val::Int(b_val) => Expr::operate_int(a_val,        b_val, op),
				Val::Float(b_val) => Expr::operate_flt(a_val as f64, b_val, op),
				_ => return false,
			},
			Val::Float(a_val) => match b {
				  Val::Int(b_val) => Expr::operate_flt(a_val, b_val as f64, op),
				Val::Float(b_val) => Expr::operate_flt(a_val, b_val,        op),
				_ => return false,
			},
			Val::Str(a_val) => match b {
				Val::Str(b_val) => Expr::operate_str(a_val, b_val, op),
				_ => return false,
			},
			_ => return false,
		};

		match res {
			None      => return false,
			Some(val) => eval_stack.push(val),
		}
		true
	}
	fn operate_int(a: i32, b: i32, op: Op) -> Option<Val<T>> {
		match op {
			Op::Add => Some(Val::Int(a + b)),
			Op::Sub => Some(Val::Int(a - b)),
			Op::Mul => Some(Val::Int(a * b)),
			Op::Div => Some(Val::Int(a / b)),
			Op::Pow if b >= 0 => Some(Val::Int(a.pow(b as uint))),
			Op::Pow if b <  0 => Some(Val::Int((a as f64).powf(b as f64) as i32)),
			Op::Mod => Some(Val::Int(a % b)),
			Op::And => Some(Val::Int(((a != 0) && (b != 0)) as i32)),
			Op::Or  => Some(Val::Int(((a != 0) || (b != 0)) as i32)),
			Op::Eq  => Some(Val::Int((a == b) as i32)),
			Op::Neq => Some(Val::Int((a != b) as i32)),
			Op::Lt  => Some(Val::Int((a <  b) as i32)),
			Op::Gt  => Some(Val::Int((a >  b) as i32)),
			Op::Leq => Some(Val::Int((a <= b) as i32)),
			Op::Geq => Some(Val::Int((a >= b) as i32)),
			_   => None,
		}
	}
	fn operate_flt(a: f64, b: f64, op: Op) -> Option<Val<T>> {
		match op {
			Op::Add => Some(Val::Float(a + b)),
			Op::Sub => Some(Val::Float(a - b)),
			Op::Mul => Some(Val::Float(a * b)),
			Op::Div => Some(Val::Float(a / b)),
			Op::Pow => Some(Val::Float(a.powf(b))),
			Op::Mod => Some(Val::Float(a % b)),
			Op::Eq  => Some(Val::Int((a == b) as i32)),
			Op::Neq => Some(Val::Int((a != b) as i32)),
			Op::Lt  => Some(Val::Int((a <  b) as i32)),
			Op::Gt  => Some(Val::Int((a >  b) as i32)),
			Op::Leq => Some(Val::Int((a <= b) as i32)),
			Op::Geq => Some(Val::Int((a >= b) as i32)),
			_   => None,
		}
	}
	fn operate_str(a: String, b: String, op: Op) -> Option<Val<T>> {
		match op {
			Op::Add => Some(Val::Str(format!("{}{}", a, b))),
			Op::Eq  => Some(Val::Int((a == b) as i32)),
			Op::Neq => Some(Val::Int((a != b) as i32)),
			_   => None,
		}
	}
}

#[cfg(test)]
mod test {
	use super::Val;

	#[test]
	fn test_math() {
		let mut form;

		form = super::parse("2 + 3 * 4").unwrap();
		assert!(form.evaluate() == Some(Val::Int(14)));

		form = super::parse("( 23.5- (22+1))/0.25").unwrap();
		assert!(form.evaluate() == Some(Val::Float(2.)));

		form = super::parse("4^3^2").unwrap();
		assert!(form.evaluate() == Some(Val::Int(262144)));

		form = super::parse("19 % 3").unwrap();
		assert!(form.evaluate() == Some(Val::Int(1)));
	}

	#[test]
	fn test_errors() {
		let mut res;

		res = super::parse("3 && 2.5");
		assert_eq!(res.unwrap().evaluate(), None);

		res = super::parse("2d4 + 6");
		assert_eq!(res.err().unwrap(), super::Error::UnknownToken("2d4".to_string()));

		res = super::parse("(6 + 2))");
		assert_eq!(res.err().unwrap(), super::Error::MismatchedParenthesis);

		res = super::parse("((6 + 2)");
		assert_eq!(res.err().unwrap(), super::Error::MismatchedParenthesis);

		res = super::parse("6 + ");
		assert_eq!(res.unwrap().evaluate(), None);

		res = super::parse("* * *");
		assert_eq!(res.unwrap().evaluate(), None);

		res = super::parse("3 ~= 2");
		assert_eq!(res.err().unwrap(), super::Error::UnknownToken("~".to_string()));
	}

	#[test]
	fn test_strings() {
		let form = super::parse(r#" "eat" + " " + "hotdogs" "#).unwrap();
		assert_eq!(form.evaluate(), Some(Val::Str("eat hotdogs".to_string())));

		assert_eq!(super::parse(r#" "eat" * 6 "#).unwrap().evaluate(), None);

		assert!(super::parse(r#" """ "#).is_err());
	}

	#[test]
	fn test_boolean() {
		let mut form;

		form = super::parse("3 == 1 + 2").unwrap();
		assert_eq!(form.evaluate(), Some(Val::Int(1)));

		form = super::parse(r#" "hotdog" != "hotdog" "#).unwrap();
		assert_eq!(form.evaluate(), Some(Val::Int(0)));

		form = super::parse("4 * 3 >= 12 && 4 > 3").unwrap();
		assert_eq!(form.evaluate(), Some(Val::Int(1)));

		form = super::parse("2 < 2 || 2 <= 2").unwrap();
		assert_eq!(form.evaluate(), Some(Val::Int(1)));

		form = super::parse("2==3||4>4").unwrap();
		assert_eq!(form.evaluate(), Some(Val::Int(0)));

		form = super::parse("(2 == 1 * 2) * 3.1 + 4.2").unwrap();
		assert!(form.evaluate().unwrap().float().unwrap() - 7.3 < 0.0001);
	}

	#[test]
	fn test_variable_setting() {		
		let mut form1 = super::parse("3 + fish * 2").unwrap();
		assert_eq!(form1.evaluate(), None);
		form1.set_variable_int("fish", 7);
		assert_eq!(form1.evaluate(), Some(Val::Int(17)));

		let mut form2 = super::parse(r#" once + upon + a + time "#).unwrap();
		form2.set_variables(|name: &str| { Val::Str(format!("{} ", name)) });
		assert_eq!(form2.evaluate(), Some(Val::Str("once upon a time ".to_string())));
	}

	/*#[test]
	fn test_function() {
		let mut parser: FormulaParser<()> = FormulaParser::new();
		parser.add_function("t10", |v| {
			if v.len() != 1 { return None; }
			match v[0] {
				Val::Int(val)   => Some(Val::Int(val * 10)),
				Val::Float(val) => Some(Val::Float(val * 10)),
				_ => return None,
			}
		});
		let form = parser.parse("5 + t10(6)");
		assert_eq!(form.evaluate(), Some(Val::Int(65)));
	}*/
}
