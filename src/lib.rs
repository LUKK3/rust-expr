#![feature(slicing_syntax)]

#![feature(phase)]
#[phase(plugin)]
extern crate regex_macros;
extern crate regex;

pub mod expr;
