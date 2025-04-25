import numpy as np

# AND回路
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# NAND回路
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# OR回路
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# XOR回路
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# Test the functions
print("AND(0, 0) =", AND(0, 0))
print("AND(0, 1) =", AND(0, 1))
print("AND(1, 0) =", AND(1, 0))
print("AND(1, 1) =", AND(1, 1))
print("NAND(0, 0) =", NAND(0, 0))
print("NAND(0, 1) =", NAND(0, 1))
print("NAND(1, 0) =", NAND(1, 0))
print("NAND(1, 1) =", NAND(1, 1))
print("OR(0, 0) =", OR(0, 0))
print("OR(0, 1) =", OR(0, 1))
print("OR(1, 0) =", OR(1, 0))
print("OR(1, 1) =", OR(1, 1))
print("XOR(0, 0) =", XOR(0, 0))
print("XOR(0, 1) =", XOR(0, 1))
print("XOR(1, 0) =", XOR(1, 0))
print("XOR(1, 1) =", XOR(1, 1))
