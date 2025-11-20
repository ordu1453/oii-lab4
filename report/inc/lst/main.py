import numpy as np
import matplotlib.pyplot as plt
import time
import pygad

np.random.seed(42)

def f(x):
    return np.sin(x) * (np.sin(x) + np.cos(x))

def g(x, a, b, c):
    return a * x**2 + b * x + c

X = np.linspace(-2*np.pi, 2*np.pi, 200)
Y = f(X)

def mse_from_params(params):
    a, b, c = params
    return np.mean((Y - g(X, a, b, c))**2)


#  Генетический алгоритм

ga_mse_history = []

def on_generation(ga_instance):
    best_sol, _, _ = ga_instance.best_solution()
    err = mse_from_params(best_sol)
    ga_mse_history.append(err)

def fitness_func(ga_instance, solution, solution_idx):
    err = mse_from_params(solution)
    return 1.0 / (err + 1e-12)

start_ga = time.time()
ga = pygad.GA(
    num_generations=1000,     
    num_parents_mating=8,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=3,
    gene_type=float,
    gene_space={'low': -5.0, 'high': 5.0},
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=33,
    random_mutation_min_val=-0.5,
    random_mutation_max_val=0.5,
    keep_parents=2,
    on_generation=on_generation,
    suppress_warnings=True
)

ga.run()
ga_time = time.time() - start_ga

ga_solution, ga_fitness, _ = ga.best_solution()
ga_mse = mse_from_params(ga_solution)

print(f"GA best params (a,b,c): {ga_solution}")
print(f"GA MSE = {ga_mse:.6e}, Time = {ga_time:.3f}s")

#  Harmony Search 

class HarmonySearch:
    def __init__(self, obj_func, num_vars, lb, ub, hms=20, hmcr=0.9, par=0.3, bw=0.1, max_iter=5000):
        self.obj_func = obj_func
        self.num_vars = num_vars
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.max_iter = max_iter

        self.HM = np.random.uniform(low=self.lb, high=self.ub, size=(self.hms, self.num_vars))
        self.HM_fitness = np.array([self.obj_func(h) for h in self.HM])
        self.best_idx = np.argmin(self.HM_fitness)
        self.best_harmony = self.HM[self.best_idx].copy()
        self.best_score = self.HM_fitness[self.best_idx]
        self.history = [self.best_score]

    def run(self):
        for it in range(self.max_iter):
            new_harmony = np.zeros(self.num_vars, dtype=float)
            for i in range(self.num_vars):
                if np.random.rand() < self.hmcr:
                    new_harmony[i] = self.HM[np.random.randint(0, self.hms), i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-self.bw, self.bw)
                else:
                    new_harmony[i] = np.random.uniform(self.lb[i], self.ub[i])
                new_harmony[i] = np.clip(new_harmony[i], self.lb[i], self.ub[i])

            new_score = self.obj_func(new_harmony)
            worst_idx = np.argmax(self.HM_fitness)
            if new_score < self.HM_fitness[worst_idx]:
                self.HM[worst_idx] = new_harmony
                self.HM_fitness[worst_idx] = new_score

            self.best_idx = np.argmin(self.HM_fitness)
            self.best_harmony = self.HM[self.best_idx].copy()
            self.best_score = self.HM_fitness[self.best_idx]
            self.history.append(self.best_score)

        return self.best_harmony, self.best_score

start_hs = time.time()
hs = HarmonySearch(
    obj_func=mse_from_params,
    num_vars=3,
    lb=[-5, -5, -5],
    ub=[5, 5, 5],
    hms=50,
    hmcr=0.9,
    par=0.3,
    bw=0.5,
    max_iter=1000
)
hs_solution, hs_mse = hs.run()
hs_time = time.time() - start_hs

print("HS best params (a,b,c):", hs_solution)
print("HS MSE:", hs_mse, "Time:", f"{hs_time:.3f}s")

#  Визуализация результатов

plt.figure(figsize=(10, 6))
plt.plot(X, Y, label="f(x) = sin(x)(sin(x)+cos(x))", linewidth=2)
plt.plot(X, g(X, *ga_solution), "--", label="GA", linewidth=2)
plt.plot(X, g(X, *hs_solution), ":", label="HS", linewidth=2)
# plt.title("Аппроксимация функции f(x) с помощью GA и Harmony Search")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(ga_mse_history, label="GA СКО", color="blue", linewidth=1.5)
plt.plot(hs.history, label="HS СКО", color="green", linewidth=1.5)
# plt.title("Сравнение сходимости: GA vs Harmony Search")
plt.xlabel("Номер итерации / поколения")
plt.ylabel("Среднеквадратичная ошибка")
plt.yscale("log")  
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Сравнение ---")
print(f"GA: MSE={ga_mse:.6e}, Time={ga_time:.3f}s")
print(f"HS: MSE={hs_mse:.6e}, Time={hs_time:.3f}s")
