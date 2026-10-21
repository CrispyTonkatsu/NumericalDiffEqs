# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solver Implementations

# %% jupyter={"source_hidden": true}
import matplotlib.pyplot as plt
import numpy as np

# +
# %matplotlib inline
# -


def euler_step(t_n, y_n, f, h):
    return y_n + h * f(t_n, y_n)


def midpoint_step(t_n, y_n, f, h):
    half_h = h / 2

    k_1 = f(t_n, y_n)
    k_2 = f(t_n + half_h, y_n + half_h * k_1)

    return y_n + h * k_2


def mod_euler_step(t_n, y_n, f, h):
    k_1 = f(t_n, y_n)
    k_2 = f(t_n + h, y_n + h * k_1)

    return y_n + (h / 2) * (k_1 + k_2)


def rk4_step(t_n, y_n, f, h):
    half_h = h / 2

    k_1 = f(t_n, y_n)
    k_2 = f(t_n + half_h, y_n + half_h * k_1)
    k_3 = f(t_n + half_h, y_n + half_h * k_2)
    k_4 = f(t_n + h, y_n + h * k_3)

    return y_n + (h / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def solve(f, solver, t_range, y_0):
    y_0 = np.atleast_1d(y_0)

    N = len(t_range)
    dim = len(y_0)

    output = np.zeros((N, dim), dtype=float)
    output[0] = y_0
    h = t_range[1] - t_range[0]

    for i in range(len(t_range) - 1):
        t = t_range[i]
        y = output[i]
        output[i + 1] = solver(t, y, f, h)

    return output


# %% [markdown]
# # 1 Dimensional ODE
# Homogeneous First Order ODE
# $$
# \dot{y} + t^{2} y = 0
# $$
# Analytical solution for $y(0) = 5$
# $$
# y = 5{e}^{-\frac{{x}^{3}}{3}}
# $$


# %% jupyter={"source_hidden": true}
def f1(t, y):
    return -(t**2) * y


def f1_sol(t):
    return 100 * np.exp((-(t**3) / (3)) - (8 / 3))


t = np.linspace(-2, 10, 300)
y = f1_sol(t)
y_0 = np.array([100])

solver_sols = [
    (solve(f1, euler_step, t, y_0), "Euler's Method"),
    (solve(f1, midpoint_step, t, y_0), "Midpoint Method"),
    (solve(f1, mod_euler_step, t, y_0), "Modified Euler's Method"),
    (solve(f1, rk4_step, t, y_0), "Runge Kutta 4"),
]

fig, ax_layout = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
fig.suptitle("$\\dot{y} + t^{2} y = 0$, for $y(-2) = 100$ and $h=12/300$")

ax_list = ax_layout.flatten()

for i, axes in enumerate(ax_list):
    y_numerical, label = solver_sols[i]
    axes.plot(t, y, label="Analytical Solution", color="blue", alpha=1.0)
    axes.plot(t, y_numerical[:, 0], label=label, color="red", alpha=0.8)

    axes.set_title(label)
    axes.set_xlabel("t")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# # System of ODEs
# System of Equations for orbit of planet with $GM = 1$:
#
# Position Equation
# $$
# \begin{bmatrix}
# x' \\ y'
# \end{bmatrix}
# =
# \begin{bmatrix}
# v_x \\ v_y
# \end{bmatrix}
# $$
#
# Velocity Equation
# $$
# \begin{bmatrix}
# v_x' \\ v_y'
# \end{bmatrix}
# =
# -\frac{1}{(x^2 + y^2)^{\frac{3}{2}}}
# \begin{bmatrix}
# x \\ y
# \end{bmatrix}
# $$
#
# Analytical Solution for (obtained [here](https://en.wikipedia.org/wiki/Eccentric_anomaly))
# $$
# \begin{bmatrix}
# x \\ y
# \end{bmatrix}
# =
# \begin{bmatrix}
# cos(t) \\ sin(t)
# \end{bmatrix}
# $$
# for
# $$
# \begin{bmatrix}
# x_0 \\ y_0 \\ vx_0 \\ vy_0
# \end{bmatrix}
# =
# \begin{bmatrix}
# 1 \\ 0 \\ 0 \\ 1
# \end{bmatrix}
# $$


# %% jupyter={"source_hidden": true}
def f2(_, state):
    x, y, vx, vy = state
    r3 = (x**2 + y**2) ** 1.5

    return np.array([vx, vy, -x / r3, -y / r3])


def f2_sol(t):
    return np.cos(t), np.sin(t)


t = np.linspace(0, 2 * np.pi, 100)
x, y = f2_sol(t)
y_0 = np.array([1, 0, 0, 1])

solver_sols = [
    (solve(f2, euler_step, t, y_0), "Euler's Method"),
    (solve(f2, midpoint_step, t, y_0), "Midpoint Method"),
    (solve(f2, mod_euler_step, t, y_0), "Modified Euler's Method"),
    (solve(f2, rk4_step, t, y_0), "Runge Kutta 4"),
]

fig, ax_layout = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
fig.suptitle("Orbital Equations for Circle")

ax_list = ax_layout.flatten()

for i, axes in enumerate(ax_list):
    solution, label = solver_sols[i]
    axes.plot(x, y, label="Analytical Solution", color="blue", alpha=1.0)
    axes.plot(solution[:, 0], solution[:, 1], label=label, color="red", alpha=0.8)

    axes.set_title(label)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Second Order ODE
# Spring mass equation
# $$
# y'' + 0.2y' + 4y = 0
# $$
#
# Analytical Solution for $y(0) = 0$ and $y'(0) = 10$
# $$
# y = \dfrac{100\,\sin\left(\frac{\sqrt{399}\,x}{10}\right)} {\sqrt{399}\,{e}^{\frac{x}{10}}}
# $$


# %% jupyter={"source_hidden": true}
def f3(_, state):
    x, vx = state

    return np.array([vx, -0.2 * vx - 4 * x])


def f3_sol(t):
    return (100 * np.sin((np.sqrt(399) * t) / (10))) / (np.sqrt(399) * np.exp(t / (10)))


t = np.linspace(0, 50, 500)
y = f3_sol(t)
y_0 = np.array([0, 10])

solver_sols = [
    (solve(f3, euler_step, t, y_0), "Euler's Method"),
    (solve(f3, midpoint_step, t, y_0), "Midpoint Method"),
    (solve(f3, mod_euler_step, t, y_0), "Modified Euler's Method"),
    (solve(f3, rk4_step, t, y_0), "Runge Kutta 4"),
]

fig, ax_layout = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
fig.suptitle("Spring Mass Equation")

ax_list = ax_layout.flatten()

for i, axes in enumerate(ax_list):
    solution, label = solver_sols[i]
    axes.plot(t, y, label="Analytical Solution", color="blue", alpha=1.0)
    axes.plot(t, solution[:, 0], label=label, color="red", alpha=0.8)

    axes.set_title(label)
    axes.set_xlabel("t")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid(True)

plt.tight_layout()
plt.show()
