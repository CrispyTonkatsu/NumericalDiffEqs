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
# The following cell has the implementation of the methods.

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
    output = np.zeros_like(t_range, dtype=float)

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

solver_sols = [
    (solve(f1, euler_step, t, 100), "Euler's Method"),
    (solve(f1, midpoint_step, t, 100), "Midpoint Method"),
    (solve(f1, mod_euler_step, t, 100), "Modified Euler's Method"),
    (solve(f1, rk4_step, t, 100), "Runge Kutta 4"),
]

fig, ax_layout = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
fig.suptitle("$\\dot{y} + t^{2} y = 0$, for $y(-2) = 100$ and $h=12/300$")

ax_list = ax_layout.flatten()

for i, axes in enumerate(ax_list):
    y_numerical, label = solver_sols[i]
    axes.plot(t, y, label="Analytical Solution", color="blue", alpha=1.0)
    axes.plot(t, y_numerical, label=label, color="red", alpha=0.8)

    axes.set_title(label)
    axes.set_xlabel("t")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid(True)

plt.tight_layout()
plt.show()

# %%
