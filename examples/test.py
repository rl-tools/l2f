from l2f import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy


# test()

device = Device()

rng = Rng()

env = Environment()
params = Parameters()
state = State()
observation = Observation()
next_state = State()
observation = Observation()
initialize_environment(device, env)
initialize_rng(device, rng, 0)




sample_initial_parameters(device, env, params, rng)
params.dynamics.mass *= 0.1
print(parameters_to_json(device, env, params))
print(f"rotor_positions: {params.dynamics.rotor_positions}")
print(f"rotor_thrust_coefficients: {params.dynamics.rotor_thrust_coefficients}")
print(f"J: {params.dynamics.J}")
print(f"J_inv: {params.dynamics.J_inv}") 
print(f"motor_time_constant: {params.dynamics.motor_time_constant}")
print(f"integration dt: {params.integration.dt}")
# sample_initial_state(device, env, params, state, rng)
initial_state(device, env, params, state)
trajectory = []
for step_i in range(100):
    action = [1, 1, 1, 1]
    print("step: ", step_i, " position", state.position, " orientation", state.orientation, " linear_velocity", state.linear_velocity, " angular_velocity", state.angular_velocity, " rpm", state.rpm)
    print(f"mass: {params.dynamics.mass}")
    step(device, env, params, state, action, next_state, rng)
    print(state.position, state.orientation, state.linear_velocity, state.angular_velocity, state.rpm)
    print("next_step: ", step_i, " position", next_state.position, " orientation", next_state.orientation, " linear_velocity", next_state.linear_velocity, " angular_velocity", next_state.angular_velocity, " rpm", next_state.rpm)
    trajectory.append(copy.copy(state))
    state.assign(next_state)
    if any(np.isnan(state.position)):
        sys.exit(1)

observe(device, env, params, state, observation, rng)


plt.plot([s.position[2] for s in trajectory])
plt.show()
