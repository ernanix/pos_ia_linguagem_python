def calcula_imposto(func,val):
    """calls the function f with 1 as its argument"""
    return func(val)


def calcula_imposto_pr(x):
    return x * 0.3

def calcula_imposto_mg(x):
    return x * 0.5



# by convention, we give classes PascalCase names
class CountingClicker:
    """A class can/should have a docstring, just like a function"""
    # these are the member functions
    # every one takes a first parameter "self" (another convention)
    def __init__(self, count = 0):
        """This is the constructor."""
        self.count = count
    def __repr__(self):
        return f"CountingClicker(count={self.count})"
    def click(self, num_times = 1):
        """Click the clicker some number of times."""
        self.count += num_times
    def read(self):
        return self.count
    def reset(self):
        self.count = 0
        

def generate_range(n):
    i = 0
    while i < n:
        #yield i # every call to yield produces a value of the generator
        i += 1

def teste(*argumentos):
    print(sum(list(argumentos)))