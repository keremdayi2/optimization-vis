import operator 

class Variable:
    def __init__(self, ndims=1):
        self.ndims = ndims
        self.value = None
    
    def __add__(self, other):
        print('adding')
        return Variable()

    def set(self, value):
        self.value = value

    def evaluate(self):
        assert(not self.value is None, 'assign values to variables before evaluating!')
        return self.value

    def __le__(self, other):
        print('le')
        return Variable()

    def __add__(self, other):
        return Expression(self, other, 'add')

    def __mul__(self, other):
        assert()

class LinearInequality:
    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2
    
    # convert to standard form inequality a^Tx <= b 
    def standard_form(self):
        pass


class Expression:
    def __init__(self, e1, e2=None, operation=None):
        self.e1 = e1
        self.e2 = e2
        self.operation = operation

    def __ge__(self, other):
        return other <= self

    def evaluate(self):
        assert(not self.operation is None)

        if self.e2 is None:
            assert(self.e1 is Variable)
            return self.operation(self.e1.evaluate())

        if self.operation == 'add':
            return self.e1.evaluate() + self.e2.evaluate()

        return self.operation(self.e1.evaluate(), self.e2.evaluate())
        # based on operation

    def __add__(self, other):
        return Expression(self, other, 'add')
