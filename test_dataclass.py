
from dataclasses import dataclass, is_dataclass


class ROOT():
    def __init__(self) -> None:
        print(f"init ROOT, {self.__class__}")

    def __post_init__(self):
        obj = super()
        assert not hasattr(obj, '__post_init__')
        print(f"post ROOT, {self.__class__}")
    
    def init_non_dataclass(self, cls):
        mro = self.__class__.mro()
        index = mro.index(cls)
        for parent in mro[index:]:    
            if not is_dataclass(parent):
                parent.__init__(self)
                break

class F(ROOT):
    def __init__(self) -> None:
        self.init_non_dataclass(ROOT)
        #super().__init__()
        print(f"init F, {self.__class__}")

@dataclass
class A(ROOT):
    a:int = 0

    def __post_init__(self):
        super().__post_init__()
        print(f"post A, {self.__class__}")

@dataclass
class B(ROOT):
    b:int = 0

    def __post_init__(self):
        super().__post_init__()
        print(f"post B, {self.__class__}")


@dataclass
class C(B):
    c:int = 0

@dataclass
class D(F):
    d:int = 0

    def __post_init__(self):
        self.init_non_dataclass(F)
        super().__post_init__()
        print(f"post D, {self.__class__}")

@dataclass
class E(C):
    def __post_init__(self):
        super().__post_init__()
        print(f"post E, {self.__class__}") 

@dataclass
class AB(A,B):
    def __post_init__(self):
        super().__post_init__()
        print(f"post AB, {self.__class__}") 

@dataclass
class AC(A,C):
    def __post_init__(self):
        super().__post_init__()
        print(f"post AC, {self.__class__}")

@dataclass
class AF(A,F):

    def __post_init__(self):
        self.init_non_dataclass(F)
        super().__post_init__()
        print(f"post AF, {self.__class__}")

@dataclass
class FA(F,A):
    def __post_init__(self):
        self.init_non_dataclass(F)
        super().__post_init__()
        print(f"post AF, {self.__class__}") 

@dataclass
class CD(C,D):
    def __post_init__(self):
        super().__post_init__()
        print(f"post CD, {self.__class__}")

@dataclass
class X():
    x:int = 0

    def __post_init__(self):
        print(f"post X, {self.__class__}")

@dataclass
class CX(C,X):
    def __post_init__(self):
        super().__post_init__()
        print(f"post CX, {self.__class__}")

@dataclass
class XC(X,C):
    def __post_init__(self):
        super().__post_init__()
        print(f"post XC, {self.__class__}") 

F()
print("-")
A()
print("-")
B()
print("-----")
C()
print("-")
D()
print("-")
E()
print("-----")
AB()
print("-")
AC()
print("-")
CD()
print("-")
print(AF.mro())
AF()
print("-")
print(FA.mro())
FA()
print("-")
try:
    CX()
    correct = False
except AssertionError as e:
    correct  = True
assert correct

XC()