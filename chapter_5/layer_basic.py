import numpy as np

# backward propagation for Multiply operation(node)
class MulLayer:
    def __init__(self):
        # save values of forward propagation, used in backward()
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    

# backward propagation for ADD operation(node)
class AddLayer:
    def __init__(self):
        pass  # actually, the variables were not needed, since there was no need to remember "intermediate variables"

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    
# simple example
apple = 100
apple_number = 2
orange = 150
orange_number = 3
tax = 1.1
# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_number)
orange_price = mul_orange_layer.forward(orange, orange_number)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
# backward
dprice = 1.0
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_number = mul_orange_layer.backward(dorange_price)
dapple, dapple_number = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_number, dapple, dorange_number, dorange, dtax)

