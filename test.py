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

print(foo)
def decorator3(func):
    print("decorator3")
    return func