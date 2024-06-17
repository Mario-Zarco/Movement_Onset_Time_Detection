import numpy as np
from math import floor


def f_tau(tau):
    return 15 * tau ** 4 - 6 * tau ** 5 - 10 * tau ** 3


def x_st(xi, xf, tau):
    return xi + (xi - xf) * f_tau(tau)


def y_st(yi, yf, tau):
    return yi + (yi - yf) * f_tau(tau)


def straight_trajectory_with_static_phase(xi, yi, xf, yf, to, tf, step_size):
    # to: onset time
    N = floor(to / step_size)
    to = N * step_size

    M = floor((tf - to) / step_size)
    tf = (N + M) * step_size

    # ---- MOVEMENT PHASE ---- #

    t_hand = np.linspace(0, tf, N + M + 1)
    t_movement = t_hand[0: M + 1]

    x_hand = []
    y_hand = []
    for t in t_movement:
        tau_ = t / (tf - to)
        x_hand = np.append(x_hand, x_st(xi, xf, tau_))
        y_hand = np.append(y_hand, y_st(yi, yf, tau_))

    # ---- STATIC PHASE ---- #
    x_hand = np.append(np.full(N, xi), x_hand)
    y_hand = np.append(np.full(N, yi), y_hand)

    return x_hand, y_hand, t_hand, to


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import derivative
    from motion_onset_detection import onset_detection

    xi_ = -0.1
    yi_ = 0.1
    xf_ = -0.9
    yf_ = 1.1
    to_ = 0.4
    tf_ = 1.2
    step = 0.01

    x, y, t, t_onsets = straight_trajectory_with_static_phase(xi_, yi_,
                                                              xf_, yf_,
                                                              to_, tf_,
                                                              step)

    fd = derivative.SavitzkyGolay(left=0.1, right=0.1, order=6, periodic=False)
    # d_dt = FinDiff(0, step_size, 1, acc=6)

    delta_T = 0.1  # 100 ms
    Ts = step  # 1 / sampling frequency
    m = int((delta_T / Ts)) - 1
    print("m = ", m)
    tm = m * Ts
    print("tm = ", tm)

    vx = fd.d(x, t)
    vy = fd.d(y, t)

    # vx = d_dt(x)
    # vy = d_dt(y)

    to_predicted, dict_results, converged, adjusted = onset_detection(m, x, y, t, vx, vy)
    print(converged, to_predicted, to_, np.abs(to_predicted - to_))

    Um = dict_results['Um']
    e_min = dict_results['min_error']
    errors = dict_results['errors']
    times = dict_results['times']
    v_max = dict_results['max_vel']
    x1 = dict_results['x1']
    y1 = dict_results['y1']
    t1 = dict_results['t1']
    x2 = dict_results['x2']
    y2 = dict_results['y2']
    t2 = dict_results['t2']

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(5, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, x, '.', label='x')
    ax.plot(t1, x1, 'mo')
    ax.plot(t2, x2, 'r.')
    ax.axvline(to_, ls='-', color='g', label='to (real)')
    ax.axvline(to_predicted, ls='--', color='r', label='to (predicted)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-0.05, t[-1] + 0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, y, '.', label='y')
    ax.plot(t1, y1, 'mo')
    ax.plot(t2, y2, 'r.')
    ax.axvline(to_, ls='--', color='g', label='to (real)')
    ax.axvline(to_predicted, ls='--', color='r', label='to (predicted)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-0.05, t[-1] + 0.05)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, vx, label='vx')
    ax.plot(t, vy, label='vy')
    ax.axhline(v_max, ls='--')
    ax.axvline(to_, ls='--', color='g', label='to (real)')
    ax.axvline(to_predicted, ls='--', color='r', label='to (predicted)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-0.05, t[-1] + 0.05)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(times, errors, '.-', label='error')
    ax.axvline(to_, ls='--', color='g', label='to (real)')
    ax.axvline(to_predicted, ls='--', color='r', label='to (predicted)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-0.05, t[-1] + 0.05)

    Um = np.array(Um)

    ax = fig.add_subplot(gs[4, 0])
    ax.plot(times, Um[:, 0], label='$U_{x}$')
    ax.plot(times, Um[:, 1], label='$U_{y}$')
    ax.axvline(to_, ls='--', color='g', label='to (real)')
    ax.axvline(to_predicted, ls='--', color='r', label='to (predicted)')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-0.05, t[-1] + 0.05)

    plt.show()