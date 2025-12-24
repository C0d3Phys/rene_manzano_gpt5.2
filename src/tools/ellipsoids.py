from dataclasses import dataclass

@dataclass(frozen=True)
class Ellipsoid:
    a: float
    b: float

    @property
    def f(self):
        return (self.a - self.b) / self.a

    @property
    def e2(self):
        return (self.a**2 - self.b**2) / self.a**2

    @property
    def ep2(self):
        return (self.a**2 - self.b**2) / self.b**2


WGS84 = Ellipsoid(
    a=6378137.0,
    b=6356752.314245
)
