# Path Integral Formulation — Całkowanie po Ścieżkach

## Wprowadzenie

**Path Integral Formulation** (sformułowanie całkowania po ścieżkach), znane też jako **całka funkcjonalna Feynmana**, to alternatywne podejście do mechaniki kwantowej opracowane przez **Richarda Feynmana** w 1948 roku, inspirowane wcześniejszą pracą Paula Diraca. Zamiast opisywać układ za pomocą równania Schrödingera, metoda ta oblicza amplitudę przejścia kwantowego jako sumę (całkę) po **wszystkich możliwych ścieżkach** łączących stan początkowy ze stanem końcowym.

### Dlaczego warto znać tę metodę?

- **Intuicyjność** — daje bezpośredni obraz kwantowej superpozycji: cząstka „próbuje" wszystkich możliwych dróg naraz.
- **Unifikacja** — łączy mechanikę kwantową z mechaniką klasyczną w jednym formalizmie.
- **Zastosowania obliczeniowe** — od kwantowej teorii pola (QFT) przez grawitację kwantową po **kwantowe uczenie maszynowe** i symulacje Monte Carlo w fizyce materii skondensowanej.
- **Metody numeryczne** — lattice QCD (siatka kwantowej chromodynamiki) i Monte Carlo po ścieżkach to narzędzia bezpośrednio oparte na tej formule.

---

## Mechanika Klasyczna — Fundament Formalizmowy

### Zasada Najmniejszego Działania

W mechanice klasycznej ruch cząstki opisuje **zasada Hamiltona**: układ ewoluuje wzdłuż ścieżki, dla której **działanie** jest stacjonarne (zwykle minimalne).

**Działanie** (ang. *action*) definiujemy jako:

```
S[q] = ∫_{t₁}^{t₂} L(q, q̇, t) dt
```

gdzie **L** to **lagranżjan** układu:

```
L = T - V = ½mq̇² - V(q)
```

Warunek stacjonarności `δS = 0` prowadzi do **równań Eulera–Lagrange'a**:

```
d/dt (∂L/∂q̇) - ∂L/∂q = 0
```

### Przykład: Oscylator Harmoniczny (klasyczny)

Dla oscylatora harmonicznego `V(q) = ½mω²q²`:

```
L = ½m(q̇² - ω²q²)
```

Równanie ruchu: `q̈ + ω²q = 0`, z rozwiązaniem `q(t) = A cos(ωt + φ).`

---

## Amplituda Przejścia w Mechanice Kwantowej

### Od Macierzy S do Całki po Ścieżkach

W mechanice kwantowej chcemy obliczyć **amplitudę przejścia** (propagator):

```
K(q_f, t_f ; q_i, t_i) = ⟨q_f | e^{-iĤ(t_f - t_i)/ℏ} | q_i⟩
```

Feynman pokazał, że propagator można wyrazić jako:

```
K(q_f, t_f ; q_i, t_i) = ∫ Dq(t) · exp(iS[q]/ℏ)
```

Symbol `∫ Dq(t)` oznacza **całkę funkcjonalną** — sumowanie po *wszystkich* ścieżkach `q(t)` spełniających warunki brzegowe `q(t_i) = q_i` oraz `q(t_f) = q_f`.

### Interpretacja Fizyczna

Każda ścieżka wnosi amplitudę `exp(iS[q]/ℏ)` — liczbę zespoloną o module 1.

- **Ścieżka klasyczna** ma działanie stacjonarne → fazy sąsiednich ścieżek konstruktywnie interferują → dominuje w granicy klasycznej (`ℏ → 0`).
- **Ścieżki dalekie od klasycznej** mają gwałtownie oscylujące fazy → znoszą się destruktywnie.

W granicy klasycznej `ℏ → 0` metoda stacjonarnej fazy odtwarza równania Eulera–Lagrange'a.

---

## Dyskretna Konstrukcja Całki po Ścieżkach

### Podział Czasu na Małe Kroki

Czas `[t_i, t_f]` dzielimy na `N` kroków `ε = (t_f - t_i)/N`:

```
K = lim_{N→∞} (m / 2πiℏε)^{N/2}
    ∫ dq₁ dq₂ … dq_{N-1}
    · exp(i/ℏ · Σₙ L_n · ε)
```

gdzie `L_n ≈ L(qₙ, (qₙ₊₁ - qₙ)/ε)`.

Każde całkowanie po `qₙ` odpowiada sumowaniu po możliwych pozycjach cząstki w chwili `tₙ`.

---

## Wolna Cząstka — Obliczenie Analityczne

### Propagator Wolnej Cząstki

Dla `V = 0` lagranżjan to `L = ½mq̇²`. Czynnik fazy dla jednego kroku:

```
exp(im(qₙ₊₁ - qₙ)² / 2ℏε)
```

Jest to gaussowska całka po pośrednich współrzędnych. Iterując N-krotnie:

```
K_0(q_f, t_f ; q_i, t_i) = sqrt(m / 2πiℏT) · exp(im(q_f - q_i)² / 2ℏT)
```

gdzie `T = t_f - t_i`. Wynik zgadza się z bezpośrednim rozwiązaniem równania Schrödingera.

### Implementacja Numeryczna w Pythonie

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# -------------------------------------------------------
# Numeryczne całkowanie po ścieżkach — wolna cząstka 1D
# Metoda: dyskretna siatka ścieżek (kratownica czasowa)
# -------------------------------------------------------

hbar = 1.0
m = 1.0

def free_particle_propagator_exact(q_f, q_i, T):
    """Dokładny propagator wolnej cząstki."""
    prefactor = np.sqrt(m / (2j * np.pi * hbar * T))
    phase = np.exp(1j * m * (q_f - q_i)**2 / (2 * hbar * T))
    return prefactor * phase

def path_integral_free_particle(q_i, q_f, T, N_steps=50, N_samples=200_000):
    """
    Szacuje propagator metodą Monte Carlo po ścieżkach.

    Parametry:
        q_i, q_f  : pozycja początkowa i końcowa
        T         : całkowity czas propagacji
        N_steps   : liczba kroków czasowych
        N_samples : liczba losowo próbkowanych ścieżek
    """
    dt = T / N_steps
    rng = np.random.default_rng(42)

    # Generowanie losowych ścieżek jako odchyleń od ścieżki liniowej
    # q(t) = q_linia(t) + η(t), η(0) = η(T) = 0
    q_line = np.linspace(q_i, q_f, N_steps + 1)  # ścieżka liniowa

    amplitude_sum = 0.0 + 0j
    for _ in range(N_samples):
        # Losowe odchylenie z warunkami brzegowymi η(0)=η(T)=0
        eta = rng.normal(0, 0.3, N_steps + 1)
        eta[0] = 0.0
        eta[-1] = 0.0

        path = q_line + eta
        velocities = np.diff(path) / dt

        # Działanie klasyczne: S = ∫ ½m ṙ² dt
        S = np.sum(0.5 * m * velocities**2) * dt

        amplitude_sum += np.exp(1j * S / hbar)

    # Normalizacja (bez absolutnej normalizacji, tylko względna)
    return amplitude_sum / N_samples

# Parametry propagacji
T_values = np.linspace(0.5, 3.0, 10)
q_i, q_f = 0.0, 1.0

exact = [free_particle_propagator_exact(q_f, q_i, T) for T in T_values]
mc    = [path_integral_free_particle(q_i, q_f, T) for T in T_values]

# Wykres modułów
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(T_values, np.abs(exact), 'b-', label='Dokładny', linewidth=2)
axes[0].plot(T_values, np.abs(mc) * np.abs(exact[0]) / (np.abs(mc[0]) + 1e-10),
             'r--', label='Monte Carlo (przeskalowany)')
axes[0].set_xlabel('Czas T')
axes[0].set_ylabel('|K(q_f, q_i, T)|')
axes[0].set_title('Propagator wolnej cząstki')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Wizualizacja przykładowych ścieżek
rng = np.random.default_rng(0)
N_steps = 50
dt_vis = 2.0 / N_steps
t_vis = np.linspace(0, 2.0, N_steps + 1)
q_line_vis = np.linspace(0, 1.0, N_steps + 1)

axes[1].set_title('Przykładowe ścieżki (N=30)')
for _ in range(30):
    eta = rng.normal(0, 0.4, N_steps + 1)
    eta[0] = eta[-1] = 0.0
    axes[1].plot(t_vis, q_line_vis + eta, alpha=0.3, linewidth=0.8, color='steelblue')
axes[1].plot(t_vis, q_line_vis, 'r-', linewidth=2, label='Ścieżka klasyczna')
axes[1].set_xlabel('Czas t')
axes[1].set_ylabel('q(t)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sciezki_feynmana.png', dpi=120)
plt.show()
```

---

## Oscylator Harmoniczny

### Propagator Kwantowy

Dla `V(q) = ½mω²q²` całka po ścieżkach jest obliczalna dokładnie. Wynik:

```
K_HO(q_f, t_f ; q_i, t_i) =
    sqrt(mω / 2πiℏ sin(ωT))
    · exp( imω / (2ℏ sin(ωT)) · [(q_f² + q_i²) cos(ωT) - 2 q_f q_i] )
```

### Odzyskanie Stanów Energetycznych

Energię stanów własnych uzyskujemy z biegunsów propagatora lub z jego rozwinięcia w szereg:

```python
import numpy as np
from scipy.special import hermite, factorial

def ho_propagator(q_f, q_i, T, omega=1.0, hbar=1.0, m=1.0, n_terms=20):
    """
    Propagator oscylatora harmonicznego jako suma po stanach własnych.

    K(q_f, T; q_i, 0) = Σ_n ψ_n(q_f) ψ_n*(q_i) exp(-iE_n T/ℏ)
    """
    result = 0.0 + 0j
    alpha = np.sqrt(m * omega / hbar)

    for n in range(n_terms):
        E_n = hbar * omega * (n + 0.5)
        # Funkcja falowa oscylatora harmonicznego
        norm = (alpha / np.pi**0.5 / 2**n / factorial(n))**0.5
        H_n = hermite(n)
        psi_f = norm * H_n(alpha * q_f) * np.exp(-0.5 * alpha**2 * q_f**2)
        psi_i = norm * H_n(alpha * q_i) * np.exp(-0.5 * alpha**2 * q_i**2)
        result += psi_f * psi_i * np.exp(-1j * E_n * T / hbar)

    return result

# Porównanie z dokładnym wzorem
def ho_exact(q_f, q_i, T, omega=1.0, hbar=1.0, m=1.0):
    """Dokładny zamknięty wzór Feynmana na propagator HO."""
    s = np.sin(omega * T)
    c = np.cos(omega * T)
    pre = np.sqrt(m * omega / (2j * np.pi * hbar * s))
    exp_arg = 1j * m * omega / (2 * hbar * s) * ((q_f**2 + q_i**2) * c - 2 * q_f * q_i)
    return pre * np.exp(exp_arg)

T_test = np.pi / 3  # unikaj T = nπ/ω (bieguny)
q_f, q_i = 1.0, 0.5

K_series = ho_propagator(q_f, q_i, T_test)
K_exact  = ho_exact(q_f, q_i, T_test)

print(f"Szereg stanów własnych:  {K_series:.6f}")
print(f"Wzór dokładny Feynmana:  {K_exact:.6f}")
print(f"Błąd względny modułu:    {abs(abs(K_series) - abs(K_exact)) / abs(K_exact):.2e}")
```

---

## Związek z Równaniem Schrödingera

Propagator spełnia równanie Schrödingera jako funkcja `(q_f, t_f)`:

```
iℏ ∂K/∂t_f = Ĥ_f K
```

z warunkiem brzegowym `K(q_f, t_f → t_i; q_i, t_i) = δ(q_f - q_i)`.

Dowód: Stosując aproksymację jednego kroku `ε → 0` w dyskretnej całce i rozwijając eksponent w szereg Taylora w `ε`, odtwarza się lewa strona równania Schrödingera. W szczególności człon kinetyczny `ℏ²/2m · ∂²/∂q²` wyłania się z gaussowskiego całkowania po odchyleniu `η = q_{n+1} - q_n`.

---

## Całki po Ścieżkach w Mechanice Statystycznej

### Przejście do Czasu Euklidesowego

Podstawiając `τ = it` (rotacja Wicka), amplituda kwantowa przechodzi w **wagę Boltzmanna**:

```
exp(iS_M/ℏ) → exp(-S_E/ℏ)
```

gdzie `S_E[q] = ∫₀^β ℏ [½m(dq/dτ)² + V(q)] dτ`  (działanie euklidesowe),
a `β = 1/(kT)` to odwrotna temperatura.

**Funkcja podziału** (ang. *partition function*) statystyczna:

```
Z = Tr[e^{-βĤ}] = ∮ Dq(τ) exp(-S_E[q]/ℏ)
```

Całkowanie obejmuje **zamknięte ścieżki** `q(0) = q(βℏ)` — pętle w czasie urojonym.

### Monte Carlo w Czasie Euklidesowym

```python
import numpy as np

def action_euclidean(path, dt, omega=1.0, m=1.0, hbar=1.0):
    """
    Dyskretne działanie euklidesowe dla oscylatora harmonicznego.
    S_E = Σ_n dt [m/2 ((q_{n+1}-q_n)/dt)² + m ω²/2 q_n²]
    """
    kin = 0.5 * m * np.sum((np.diff(path) / dt)**2) * dt
    pot = 0.5 * m * omega**2 * np.sum(path[:-1]**2) * dt
    return (kin + pot) / hbar

def metropolis_path_integral(beta=4.0, N=100, n_sweeps=50_000, omega=1.0):
    """
    Algorytm Metropolisa dla całki po ścieżkach w czasie euklidesowym.
    Estymuje ⟨q²⟩ = ⟨x̂²⟩ → kT/(mω²) w granicy klasycznej.
    """
    dt = beta / N
    path = np.zeros(N + 1)   # zamknięta ścieżka: path[0] = path[N]
    path[N] = path[0]

    rng = np.random.default_rng(0)
    S = action_euclidean(path, dt, omega)
    
    q2_sum = 0.0
    accepted = 0

    for sweep in range(n_sweeps):
        for i in range(1, N):
            old_q = path[i]
            new_q = old_q + rng.uniform(-0.5, 0.5)

            # Zmiana działania (tylko lokalna: zależy od sąsiadów i-1, i, i+1)
            dS = (
                0.5 * m * ((new_q - path[i-1])**2 - (old_q - path[i-1])**2) / dt
              + 0.5 * m * ((path[i+1] - new_q)**2 - (path[i+1] - old_q)**2) / dt
              + 0.5 * m * omega**2 * (new_q**2 - old_q**2) * dt
            ) / hbar

            if dS < 0 or rng.random() < np.exp(-dS):
                path[i] = new_q
                accepted += 1

        # Warunek periodyczny
        path[N] = path[0]
        q2_sum += np.mean(path[:-1]**2)

    q2_mean = q2_sum / n_sweeps

    # Kwantowe oczekiwanie ⟨q²⟩ = ℏ/(2mω) · coth(βℏω/2)
    q2_quantum = hbar / (2 * m * omega) / np.tanh(beta * hbar * omega / 2)

    print(f"Monte Carlo ⟨q²⟩  = {q2_mean:.4f}")
    print(f"Dokładne ⟨q²⟩     = {q2_quantum:.4f}")
    print(f"Wskaźnik akceptacji: {accepted / (n_sweeps * (N-1)):.2%}")
    return q2_mean

m = 1.0
metropolis_path_integral(beta=2.0, N=80, omega=1.0)
```

---

## Kwantowa Teoria Pola — Całki Funkcjonalne

### Lagranżjan Pola Skalarnego

W QFT zmienna `q(t)` zastępowana jest **polem** `φ(x, t)` zależnym od czterech współrzędnych czasoprzestrzennych `xᵘ = (t, x, y, z)`. Działanie:

```
S[φ] = ∫ d⁴x L(φ, ∂_μ φ)
```

Dla teorii `λφ⁴`:

```
L = ½(∂_μ φ)² - ½m²φ² - λ/4! φ⁴
```

**Całkowa funkcja generująca:**

```
Z[J] = ∫ Dφ exp( i/ℏ (S[φ] + ∫ d⁴x J(x)φ(x)) )
```

Funkcje korelacji uzyskujemy przez różniczkowanie funkcjonalne:

```
⟨φ(x₁) … φ(xₙ)⟩ = (1/Z) · (-iℏ)ⁿ δⁿZ/δJ(x₁)…δJ(xₙ) |_{J=0}
```

### Diagramy Feynmana jako Rozwinięcie Perturbacyjne

Rozwijając `exp(iS_int/ℏ)` w szereg potęgowy w `λ`, każdy wyraz odpowiada **diagramowi Feynmana**:

```
⟨φ(x₁)φ(x₂)⟩ = ──────────────── + ──────⊗────── + …
               propagator wolny      człon masy
```

Reguły Feynmana (w przestrzeni pędów):
- Propagator: `i/(p² - m² + iε)`
- Wierzchołek `λφ⁴`: `-iλ`
- Pętla: całkowanie po pędzie wewnętrznym `∫ d⁴k/(2π)⁴`

---

## Zastosowania Numeryczne — Lattice QCD

### Dyskretyzacja na Siatce Euklidesowej

W **Lattice QCD** (siatkowej kwantowej chromodynamice) pola kwarków i gluonów umieszcza się na czterowymiarowej dyskretnej siatce `aⁿ` o stałej siatki `a`. Działanie Wilsona:

```
S_W = β Σ_{plaquettes} [1 - Re Tr(U_plaquette) / Nc]
```

gdzie `U` to macierze unitarne (linki gluonowe), a `β = 2Nc/g²`.

Masy hadronów oblicza się metodą **Monte Carlo HMC** (Hybrid Monte Carlo):

```python
import numpy as np

class LatticeHMC:
    """
    Uproszczona demonstracja Hybrid Monte Carlo dla teorii φ⁴ na siatce 1D.
    Prawa strona: S[φ] = Σ_x [½(∂φ)² + ½m²φ² + λφ⁴]
    """

    def __init__(self, N=20, m2=1.0, lam=0.1, dt=0.05, n_leapfrog=10):
        self.N = N
        self.m2 = m2
        self.lam = lam
        self.dt = dt
        self.n_leapfrog = n_leapfrog
        self.phi = np.zeros(N)
        self.rng = np.random.default_rng(42)

    def action(self, phi):
        kin = 0.5 * np.sum((np.roll(phi, -1) - phi)**2)
        mass = 0.5 * self.m2 * np.sum(phi**2)
        inter = self.lam * np.sum(phi**4)
        return kin + mass + inter

    def force(self, phi):
        """F = -dS/dφ"""
        lap = np.roll(phi, -1) - 2*phi + np.roll(phi, 1)
        return lap - self.m2 * phi - 4 * self.lam * phi**3

    def leapfrog(self, phi, pi):
        phi = phi.copy()
        pi = pi.copy()
        pi += 0.5 * self.dt * self.force(phi)
        for _ in range(self.n_leapfrog - 1):
            phi += self.dt * pi
            pi += self.dt * self.force(phi)
        phi += self.dt * pi
        pi += 0.5 * self.dt * self.force(phi)
        return phi, pi

    def step(self):
        pi = self.rng.standard_normal(self.N)
        H_old = self.action(self.phi) + 0.5 * np.sum(pi**2)
        phi_new, pi_new = self.leapfrog(self.phi, pi)
        H_new = self.action(phi_new) + 0.5 * np.sum(pi_new**2)
        dH = H_new - H_old
        if dH < 0 or self.rng.random() < np.exp(-dH):
            self.phi = phi_new
            return True
        return False

    def run(self, n_steps=2000, n_therm=500):
        accepted = 0
        phi2_list = []
        for i in range(n_steps):
            acc = self.step()
            if i >= n_therm:
                accepted += acc
                phi2_list.append(np.mean(self.phi**2))
        acc_rate = accepted / (n_steps - n_therm)
        phi2_mean = np.mean(phi2_list)
        print(f"⟨φ²⟩ = {phi2_mean:.4f}  |  Akceptacja: {acc_rate:.2%}")
        return phi2_list

hmc = LatticeHMC(N=20, m2=1.0, lam=0.1)
phi2_samples = hmc.run(n_steps=3000, n_therm=500)
```

---

## Całki po Ścieżkach w Uczeniu przez Wzmacnianie

### Path Integral Policy Gradient (PI²)

Metoda **PI²** (Path Integral Policy Improvement) wyznacza optymalną politykę sterowania robotem bez modelu nagrody, stosując wzór Feynmana–Kaca:

```
u*(t) = argmin E[∫_t^T (q(s) + ½||u(s)||²_R) ds + φ(x_T)]
```

Optymalną politykę można wyrazić jako **ważoną sumę po trajektoriach**:

```
u*(t, x) = Σ_k w_k(x) · ε_k(t)
```

gdzie wagi `wₖ ∝ exp(-S_k/λ)` zależą od koszt-to-go S_k każdej trajektorii.

```python
import numpy as np

def pi2_controller(dynamics_fn, cost_fn, x0, T=2.0, dt=0.05,
                   n_rollouts=200, lam=0.1, sigma=1.0):
    """
    Uproszczona implementacja PI² dla systemu 1D.

    Parametry:
        dynamics_fn : x_{t+1} = dynamics_fn(x_t, u_t) + σ·ε
        cost_fn     : kost terminalna φ(x_T)
        x0          : stan początkowy
        T           : horyzont czasu
        dt          : krok czasowy
        n_rollouts  : liczba próbkowanych trajektorii
        lam         : temperatura (analogia do ℏ)
        sigma       : odchylenie standardowe zakłóceń sterowania
    """
    n_steps = int(T / dt)
    u_baseline = np.zeros(n_steps)  # bazowe sterowanie
    rng = np.random.default_rng(42)

    all_costs = np.zeros(n_rollouts)
    all_noise = np.zeros((n_rollouts, n_steps))

    # Próbkowanie trajektorii
    for k in range(n_rollouts):
        x = x0
        noise = rng.normal(0, sigma, n_steps)
        all_noise[k] = noise
        total_cost = 0.0

        for t in range(n_steps):
            u = u_baseline[t] + noise[t]
            x = dynamics_fn(x, u, dt)
            total_cost += 0.5 * (noise[t]**2 / sigma**2) * dt  # koszt sterowania

        all_costs[k] = total_cost + cost_fn(x)

    # Wagi Feynmana–Kaca
    min_cost = np.min(all_costs)
    weights = np.exp(-(all_costs - min_cost) / lam)
    weights /= np.sum(weights)

    # Aktualizacja sterowania
    u_new = np.zeros(n_steps)
    for t in range(n_steps):
        u_new[t] = u_baseline[t] + np.sum(weights * all_noise[:, t])

    return u_new, weights

# Przykład: wahadło odwrócone (linearyzacja)
def pendulum_dynamics(x, u, dt):
    """Stan: [θ, θ̇]. Linearyzacja wokół θ=π."""
    A = np.array([[0, 1], [9.8, 0]])  # niestabilny
    return x + A @ x * dt + np.array([0, u * dt])

def terminal_cost(x):
    return 10.0 * x[0]**2 + x[1]**2

x0 = np.array([0.1, 0.0])
u_opt, w = pi2_controller(pendulum_dynamics, terminal_cost, x0)
print(f"Optymalne u(0) = {u_opt[0]:.4f}")
print(f"Entropia wag: {-np.sum(w * np.log(w + 1e-10)):.3f} (max = {np.log(200):.3f})")
```

---

## Metoda Stacjonarnej Fazy

### Aproksymacja WKB

Dla dużych `S/ℏ` dominuje ścieżka klasyczna `q_cl(t)`. Rozwijamy wokół niej:

```
q(t) = q_cl(t) + η(t)
S[q] = S[q_cl] + ½ ∫∫ η(t) δ²S/δq(t)δq(t') η(t') dt dt' + O(η³)
```

Całka po odchyleniach `η(t)` jest gaussowska:

```
K ≈ A(t_f, t_i) · exp(iS[q_cl]/ℏ)
```

gdzie amplituda preeksponencjalna `A` to wyznacznik funkcjonalny (czynnik Van Vlecka–Pauli–Morette).

### Granica Klasyczna

W granicy `ℏ → 0` metoda stacjonarnej fazy odtwarza **zasadę Hamiltona** mechaniki klasycznej — piękna demonstracja, że mechanika klasyczna wyłania się z kwantowej.

---

## Efekty Topologiczne — Instantony

### Konfiguracje Niesupresowane

W teorii z degenerowanymi minimami potencjału `V(q) = λ(q² - a²)²` pojawiają się rozwiązania interpolujące zwane **instantonami**:

```
q_inst(τ) = a · tanh(√(2λ) a (τ - τ₀))
```

Działanie instantonu: `S_inst = 4/3 · √(2λ) · a³`

Efekty tunelowania: rozszczepienie energii między stanami symetrycznymi i antysymetrycznymi:

```
ΔE = ℏω₀ · C · exp(-S_inst/ℏ)
```

gdzie `C` to czynnik preeksponencjalny z kwantowych fluktuacji wokół instantonu.

---

## Podsumowanie i Porównanie Formalizmów

| Cecha | Schrödinger | Heisenberg | Feynman (ścieżki) |
|-------|-------------|------------|-------------------|
| Zmienna centralna | Funkcja falowa ψ(q, t) | Operatory Â(t) | Propagator K(q_f, q_i, T) |
| Równanie podstawowe | iℏ ∂ψ/∂t = Ĥψ | dÂ/dt = i[Ĥ, Â]/ℏ | K = ∫Dq exp(iS/ℏ) |
| Intuicja | Fala prawdopodobieństwa | Obserwable ewoluują | Suma po historiach |
| Granica klasyczna | Pakiet falowy → cząstka | Zasada korespondencji | Metoda stacjonarnej fazy |
| Najsilniejsze zastosowanie | 1-cząstkowe układy | Kwantowa optyka | QFT, statystyka, RL |

---

## Powiązane Artykuły

- [Deep Learning](#deep-learning)
- [Uczenie przez Wzmacnianie](#reinforcement-learning)
- [Teoria Sterowania](#control-theory)
- [Optymalizacja Trajektorii](#trajectory-optimization)

---

## Zasoby

- R. Feynman, A. Hibbs — *Quantum Mechanics and Path Integrals* (1965) — klasyczna monografia
- M. Peskin, D. Schroeder — *An Introduction to Quantum Field Theory* — rozdz. 9 (całki funkcjonalne)
- J. Zinn-Justin — *Quantum Field Theory and Critical Phenomena* — zastosowania w fizyce statystycznej
- [Feynman Lectures on Physics Vol. III](https://www.feynmanlectures.caltech.edu/) — dostępne bezpłatnie
- [Path Integrals in Physics — Chaichian, Demichev](https://arxiv.org/abs/hep-th/9811014) — przegląd zastosowań
- [PI² — Theodorou et al. 2010](https://arxiv.org/abs/1011.1555) — zastosowanie w robotyce

---

*Ostatnia aktualizacja: 2025-01-01*  
*Autor: Zespół AI Robot Lab*
