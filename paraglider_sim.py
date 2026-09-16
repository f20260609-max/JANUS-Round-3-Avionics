import numpy as np
import matplotlib.pyplot as plt

# Physical & identified aerodynamic parameters (Tables 1, 2, 3)
MASS = 5.0              # Total mass m (kg)
SPAN = 3.0              # Canopy span b (m)
AREA = 3.0              # Canopy area S (m^2)
RHO = 1.225             # Air density (kg/m^3)

I_XX = 2.5              # Roll inertia (kg*m^2)
I_ZZ = 3.2              # Yaw inertia (kg*m^2)

# Trim gliding velocities
U_TRIM = 6.2            # Forward speed (m/s)
W_TRIM = 1.92           # Descent sink rate (m/s)
V_AIR = np.hypot(U_TRIM, W_TRIM)
Q_DYN = 0.5 * RHO * (V_AIR ** 2)

# Identified stability coefficients
C_L_PHI   = -0.085
C_L_P     = -0.160
C_L_DA    =  0.0025
C_N_R     = -0.140
C_N_DA    = -0.022


def parafoil_derivatives(state, delta_a, wind=np.zeros(3)):
    x, y, z, phi, psi, p, r = state
    dim_moment = Q_DYN * AREA * SPAN
    b_norm = SPAN / (2.0 * V_AIR)

    # Roll & yaw aerodynamic moments
    L_a = dim_moment * (C_L_PHI * phi + C_L_P * (p * b_norm) + C_L_DA * delta_a)
    N_a = dim_moment * (C_N_R * (r * b_norm) + C_N_DA * delta_a)

    p_dot = L_a / I_XX
    r_dot = N_a / I_ZZ
    phi_dot = p
    psi_dot = r

    # Translational kinematics with wind
    x_dot = U_TRIM * np.cos(psi) + wind[0]
    y_dot = U_TRIM * np.sin(psi) + wind[1]
    z_dot = -W_TRIM + wind[2]

    return np.array([x_dot, y_dot, z_dot, phi_dot, psi_dot, p_dot, r_dot])


def compute_control(x, y, psi, target=(0.0, 0.0), kp=1.8, kd=0.6, r=0.0):
    dx = target[0] - x
    dy = target[1] - y
    desired_psi = np.arctan2(dy, dx)
    psi_err = (desired_psi - psi + np.pi) % (2.0 * np.pi) - np.pi
    delta_cmd = -kp * psi_err - kd * r
    return np.clip(delta_cmd, -1.0, 1.0)


def run_simulation(wind=np.array([0.0, 0.0, 0.0]), dt=0.05):
    x0 = np.random.uniform(-500.0, 500.0)
    y0 = np.random.uniform(-500.0, 500.0)
    z0 = 700.0  # Spawn height = 700m

    state = np.array([x0, y0, z0, 0.0, np.random.uniform(0, 2*np.pi), 0.0, 0.0])
    trajectory = [state.copy()]
    times = [0.0]
    t = 0.0

    while state[2] > 0.0 and t < 600.0:
        delta_a = compute_control(state[0], state[1], state[4], r=state[6])
        k1 = parafoil_derivatives(state, delta_a, wind)
        k2 = parafoil_derivatives(state + 0.5 * dt * k1, delta_a, wind)
        k3 = parafoil_derivatives(state + 0.5 * dt * k2, delta_a, wind)
        k4 = parafoil_derivatives(state + dt * k3, delta_a, wind)
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        t += dt
        times.append(t)
        trajectory.append(state.copy())

    return np.array(times), np.array(trajectory)


if __name__ == "__main__":
    times, traj = run_simulation()
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    miss = np.hypot(x[-1], y[-1])

    print(f"Touchdown Time: {times[-1]:.1f} s")
    print(f"Touchdown Point: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")
    print(f"Target Error: {miss:.2f} m")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="navy", lw=2, label="Paraglider Path")
    ax.scatter(x[0], y[0], z[0], color="green", s=50, label=f"Spawn (Z={z[0]:.0f}m)")
    ax.scatter(0, 0, 0, color="red", marker="X", s=80, label="Target (0,0,0)")
    ax.scatter(x[-1], y[-1], z[-1], color="orange", s=50, label=f"Landing ({miss:.1f}m error)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude Z (m)")
    ax.set_title("Autonomous Paraglider Descent to (0,0,0)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()