class Value:
    """Stores one scalar number."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


a = Value(5.0)
b = Value(-2.0)
b.data += 0.001 
c = a * b
d = c + b
e = c * d
e.grad = 1.0
e._backward()
d._backward()
c._backward()
b._backward()
a._backward()
print(e , 'e')
print(d , 'd')
print(c , 'c')
print(b , 'b')
print(a , 'a')