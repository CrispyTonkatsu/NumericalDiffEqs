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
