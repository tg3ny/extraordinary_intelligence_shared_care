"""
Early birth at small versus large brain size, under private and shared care.

Reproduces the survival values quoted in the fossil section of the manuscript
(the Homo floresiensis discussion). In the original model (A = 0), early birth
pays off only at large brain size, so a lineage whose brain shrank while birth
stayed early would sit far below the fitness ridge. Under shared care (A = 8,
the value used in Fig. 2D), early birth with a small brain is nearly as fit as
early birth with a large one.

Values are survival to adulthood (birth survival x childhood survival) along
the linked axis I = R at birth age T = 4 in model units.
"""
from pk_model import fitness

T = 4.0
cases = {}
for A in (0.0, 8.0):
    for R in (2.0, 5.0):
        cases[(A, R)] = float(fitness(T, R, Ip=R, allocare=A))

print(f"Survival to adulthood at T = {T:g} (I = R):")
for (A, R), v in cases.items():
    print(f"  A = {A:g}, R = {R:g}: {v:.3f}")

# Values reported in the manuscript (rounded to two decimals)
assert abs(cases[(0.0, 2.0)] - 0.04) < 0.005, cases[(0.0, 2.0)]
assert abs(cases[(0.0, 5.0)] - 0.29) < 0.005, cases[(0.0, 5.0)]
assert abs(cases[(8.0, 2.0)] - 0.53) < 0.005, cases[(8.0, 2.0)]
assert abs(cases[(8.0, 5.0)] - 0.62) < 0.005, cases[(8.0, 5.0)]

ratio_private = cases[(0.0, 5.0)] / cases[(0.0, 2.0)]
ratio_shared = cases[(8.0, 5.0)] / cases[(8.0, 2.0)]
print(f"Large-brain advantage at early birth: {ratio_private:.1f}x under private care, "
      f"{ratio_shared:.2f}x under shared care (A = 8).")
assert ratio_private > 6 and ratio_shared < 1.2
print("OK: values match the manuscript.")
