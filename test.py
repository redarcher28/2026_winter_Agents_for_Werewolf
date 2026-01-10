def decorator1(func):
    print("decorator1")
    return func
def decorator2(func):
    print("decorator2")
    return func
@decorator1
@decorator2
def foo():
    pass
foo()
print("world")
print("hello world")
print("diwoefijwerf")