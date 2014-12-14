enum Func1<'a, T> {
	Arg0(         ||:'a -> T),
	Arg1(        |T|:'a -> T),
	Arg2(      |T,T|:'a -> T),
	Arg3(    |T,T,T|:'a -> T),
	Arg4(  |T,T,T,T|:'a -> T),
	Arg5(|T,T,T,T,T|:'a -> T),
}
enum Func2<'a, A, B> {
	Arg0(|A|:'a     -> B),
	Arg1(|A,B|:'a   -> B),
	Arg2(|A,B,B|:'a -> B),
}
enum FuncC<'a, A, B> {
	Arg1(      |A|:'a -> B),
	Arg2(    |A,A|:'a -> B),
	Arg3(  |A,A,A|:'a -> B),
	Arg4(|A,A,A,A|:'a -> B),
}
enum Func<'a, T> {
	  Any(Func1<'a, Val<T>),
	  Int(Func1<'a, i32>),
	Float(Func1<'a, f64>),
	 Cust(Func1<'a, T>),
	StrFloat(Func2<'a, &str, f64),
	StrInt(  Func2<'a, &str, i32),
	IntFloat(Func2<'a, i32, f64),
	FloatInt(Func2<'a, f64, i32),
	CustInt(  Func2<'a, T, i32),
	CustFloat(Func2<'a, T, f64),
	IntCust(  FuncC<'a, i32, T),
	FloatCust(FuncC<'a, f64, T),
}