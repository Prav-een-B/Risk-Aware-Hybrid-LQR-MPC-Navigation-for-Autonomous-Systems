"""Fix run_simulation.py MPC parameters"""
filepath = "run_simulation.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the MPC constructor in run_mpc_simulation
old = '''    mpc = MPCController(
        horizon=15,                       # Increased: 6→15 for better obstacle look-ahead
        Q_diag=[15.0, 15.0, 50.0],       # Aggressive heading tracking
        R_diag=[0.1, 0.1],
        P_diag=[30.0, 30.0, 40.0],       # Strong terminal heading
        d_safe=0.3,
        slack_penalty=50000.0,            # 10x stricter: 5000→50000
        slack_penalty_l1=20000.0,         # L1 penalty prevents cheap small violations
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        solver='OSQP',
        block_size=1,                     # No move-blocking: allows agile maneuvers
        w_max=0.05,                       # Tube MPC: disturbance bound (5cm)
        sqp_iterations=3                  # SQP re-linearization iterations
    )'''

new = '''    mpc = MPCController(
        horizon=8,
        Q_diag=[15.0, 15.0, 50.0],
        R_diag=[0.1, 0.1],
        P_diag=[20.0, 20.0, 40.0],
        d_safe=0.3,
        slack_penalty=10000.0,
        slack_penalty_l1=5000.0,
        dt=dt,
        v_max=2.0,
        omega_max=3.0,
        solver='OSQP',
        block_size=1,
        w_max=0.05
    )'''

if old in content:
    content = content.replace(old, new, 1)
    print("Replaced MPC constructor in run_mpc_simulation")
else:
    # Try with \r\n
    old_crlf = old.replace('\n', '\r\n')
    new_crlf = new.replace('\n', '\r\n')
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf, 1)
        print("Replaced MPC constructor (CRLF)")
    else:
        print("ERROR: Could not find MPC constructor")
        exit(1)

# Fix mpc_rate
old_rate = "    mpc_rate = 2  # Run MPC every 2 steps for tight response"
new_rate = "    mpc_rate = 3  # Run MPC every 3 steps"
content = content.replace(old_rate, new_rate)

# Also fix comparison MPC constructor
old_cmp = '''    mpc = MPCController(horizon=15, 
                       Q_diag=[15.0, 15.0, 50.0],
                       R_diag=[0.1, 0.1], 
                       P_diag=[30.0, 30.0, 40.0],
                       d_safe=0.3, slack_penalty=50000.0,
                       slack_penalty_l1=20000.0, dt=dt, 
                       v_max=2.0, omega_max=3.0, solver='OSQP',
                       sqp_iterations=3)'''

new_cmp = '''    mpc = MPCController(horizon=8, 
                       Q_diag=[15.0, 15.0, 50.0],
                       R_diag=[0.1, 0.1], 
                       P_diag=[20.0, 20.0, 40.0],
                       d_safe=0.3, slack_penalty=10000.0,
                       slack_penalty_l1=5000.0, dt=dt, 
                       v_max=2.0, omega_max=3.0, solver='OSQP')'''

if old_cmp in content:
    content = content.replace(old_cmp, new_cmp, 1)
    print("Replaced comparison MPC constructor")
else:
    old_cmp_crlf = old_cmp.replace('\n', '\r\n')
    new_cmp_crlf = new_cmp.replace('\n', '\r\n')
    if old_cmp_crlf in content:
        content = content.replace(old_cmp_crlf, new_cmp_crlf, 1)
        print("Replaced comparison MPC constructor (CRLF)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")
