"""Learning `self`: a five-step lesson, then a six-task quiz.

    python x/learn_self.py

Use the switches just below to control what prints.
"""

# ========================================================================
#  SWITCHES - edit these two lines
# ========================================================================

SHOW_LESSON = -1         # 0 runs every step. 1..5 for just one. -1 skips the lesson.
RUN_TASK = 6            # 0 runs every task. 1..6 for just one. -1 skips the quiz.

# ========================================================================


def step(n):
    """True when this lesson step should print."""
    return SHOW_LESSON == 0 or SHOW_LESSON == n


def show(task):
    """True when this quiz task should print."""
    return RUN_TASK == 0 or RUN_TASK == task


def check(label, got, want):
    ok = "PASS" if got == want else "FAIL"
    print(f"  [{ok}] {label}   got {got!r}   want {want!r}")


# ========================================================================
#  THE LESSON
# ========================================================================

def decompose_plain(W, psi, A):
    return W - A * W ** psi


class Material:
    def __init__(self, psi, A):
        self.psi = psi            # WRITE: put psi on this object
        self.A = A

    def decompose(self, W):
        return W - self.A * W ** self.psi     # READ: get them back


class Broken:
    def __init__(self):
        self.psi = 3.0

    def works(self):
        return self.psi

    def fails(self):
        return psi                # no self -> python searches the room


tacot_demo = Material(psi=3.0, A=0.01)
fast = Material(psi=3.0, A=0.50)
slow = Material(psi=3.0, A=0.01)
b = Broken()

if step(1):
    print("STEP 1  no class, pass everything every time")
    print("  decompose_plain(0.8, 3.0, 0.01) =", decompose_plain(0.8, 3.0, 0.01))
    print("  decompose_plain(0.6, 3.0, 0.01) =", decompose_plain(0.6, 3.0, 0.01))
    print("  -> psi and A repeated on every call. easy to get wrong.")
    print()

if step(2):
    print("STEP 2  a class remembers")
    print("  tacot.decompose(0.8) =", tacot_demo.decompose(0.8))
    print("  tacot.decompose(0.6) =", tacot_demo.decompose(0.6))
    print("  -> psi and A are stored. you only pass what changes.")
    print()

if step(3):
    print("STEP 3  self says WHICH object")
    print("  fast.decompose(0.8) =", fast.decompose(0.8))
    print("  slow.decompose(0.8) =", slow.decompose(0.8))
    print("  same method, same input, different answer")
    print("  fast.A =", fast.A, "   slow.A =", slow.A)
    print()
    print("  when decompose runs on fast, self IS fast")
    print("  when decompose runs on slow, self IS slow")
    print()

if step(4):
    print("STEP 4  what an object actually is")
    print("  fast.__dict__ =", fast.__dict__)
    print("  slow.__dict__ =", slow.__dict__)
    print("  self.A is a lookup of 'A' in that dict")
    print()
    print("  fast.decompose(0.8)            =", fast.decompose(0.8))
    print("  Material.decompose(fast, 0.8)  =", Material.decompose(fast, 0.8))
    print("  -> identical. the dot just passes the object in as self.")
    print()

if step(5):
    print("STEP 5  why you cannot drop self")
    print("  b.works() =", b.works())
    try:
        b.fails()
    except NameError as e:
        print("  b.fails() -> NameError:", e)
    print("  psi lives INSIDE b. python never looks inside objects on its own.")
    print()


# ========================================================================
#  QUIZ
# ------------------------------------------------------------------------
#  Set RUN_TASK at the top to work one task at a time.
#  Every task prints PASS or FAIL. Ask me for help on any of them.
# ========================================================================

print("=" * 60)
print(f"QUIZ    (RUN_TASK = {RUN_TASK}, 0 means all)")
print("=" * 60)


# ---------------------------------------------------------------- TASK 1
# PREDICT, then run. Do not change the code.
# Write your prediction in the blank BEFORE running.
#
#   my prediction for q.value is: _____25_____

class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self):
        self.value = self.value + 10


if show(1):
    q = Counter(5)
    q.bump()
    q.bump()
    print("\nTASK 1  predict the output")
    print("  q.value =", q.value)


# ---------------------------------------------------------------- TASK 2
# FIX IT. This class raises NameError. Change ONE line.

class Thermo:
    def __init__(self, temp_k):
        self.temp_k = temp_k

    def to_celsius(self):
        return self.temp_k - 273.15          # <-- TASK 2: fix this line


if show(2):
    print("\nTASK 2  fix the NameError")
    try:
        t = Thermo(373.15)
        check("Thermo(373.15).to_celsius()", round(t.to_celsius(), 2), 100.0)
    except NameError as e:
        print("  [FAIL] still broken ->", e)


# ---------------------------------------------------------------- TASK 3
# WRITE __init__ yourself.
# A Cell should remember its temperature and its density.

class Cell:
    def __init__(self, T, rho):
        self.T = T
        self.rho = rho


if show(3):
    print("\nTASK 3  write __init__")
    try:
        c = Cell(T=800.0, rho=280.0)
        check("c.T", c.T, 800.0)
        check("c.rho", c.rho, 280.0)
    except AttributeError as e:
        print("  [FAIL] nothing stored ->", e)


# ---------------------------------------------------------------- TASK 4
# WRITE A METHOD.
# Add a method called is_hot that returns True when the cell's own
# temperature is above 700 K, and False otherwise.
# It takes no arguments except self.

class Cell2:
    def __init__(self, T, rho):
        self.T = T
        self.rho = rho

    def is_hot(self):
        return self.T > 700

if show(4):
    print("\nTASK 4  write a method")
    try:
        check("Cell2(800, 280).is_hot()", Cell2(800.0, 280.0).is_hot(), True)
        check("Cell2(300, 280).is_hot()", Cell2(300.0, 280.0).is_hot(), False)
    except AttributeError as e:
        print("  [FAIL] no such method ->", e)


# ---------------------------------------------------------------- TASK 5
# TWO OBJECTS. No code to write, just answer.
# How many separate copies of T exist in memory?
#
#   my answer: ______0______

if show(5):
    virgin = Cell2(T=300.0, rho=280.0)
    charred = Cell2(T=1400.0, rho=220.0)
    print("\nTASK 5  how many copies of T exist?")
    print("  virgin.__dict__  =", virgin.__dict__)
    print("  charred.__dict__ =", charred.__dict__)


# ---------------------------------------------------------------- TASK 6
# THE REAL PATTERN.
# This mirrors what HeatConductanceMPGNN does in
# LAUMeltFlow/heat_python/gnn/gnn_flux_model.py
#
# Species stores a LIST of virgin densities and a LIST of char densities,
# then computes the virgin fraction W for given current densities.
#
#   W = (rho_now - rho_char) / (rho_virgin - rho_char)
#
# virgin_fraction takes a list of current densities, one per species,
# and returns a list of W values.

class Species:
    def __init__(self, rho_virgin, rho_char):
        self.rho_virgin = rho_virgin
        self.rho_char = rho_char

    def virgin_fraction(self, rho_now):
        W = []
        for i in range(len(rho_now)):
            span = self.rho_virgin[i] - self.rho_char[i]
            w = (rho_now[i] - self.rho_char[i]) / span
            W.append(w)
        return W
            


if show(6):
    print("\nTASK 6  the real pattern")
    try:
        tacot = Species(rho_virgin=[300.0, 900.0], rho_char=[0.0, 600.0])
        check("all virgin", tacot.virgin_fraction([300.0, 900.0]), [1.0, 1.0])
        check("all charred", tacot.virgin_fraction([0.0, 600.0]), [0.0, 0.0])
        check("halfway", tacot.virgin_fraction([150.0, 750.0]), [0.5, 0.5])
    except (AttributeError, TypeError) as e:
        print("  [FAIL] not finished ->", e)

print()
print("=" * 60)
