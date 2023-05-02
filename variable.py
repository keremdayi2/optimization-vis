class Variable:
    def __init__(self, ndims=1):
        self.ndims = ndims
    
    def __add__(self, other):
        print('adding')
        return Variable()

    def __le__(self, other):
        print('le')
        return Variable()

class Expression:
    def __init__(self):
        self.e1 = ' '
        self.e2 = None
    
    def evaluate(self, x):
        if self.e1

x = Variable()

y = x + 5

print(x <= 5)