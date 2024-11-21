from l2f import *
from l2f import vector1024 as vector
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy
import time


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
single_env_params_json = parameters_to_json(device, env, params)
print(single_env_params_json)
print(f"rotor_positions: {params.dynamics.rotor_positions}")
print(f"rotor_thrust_coefficients: {params.dynamics.rotor_thrust_coefficients}")
print(f"J: {params.dynamics.J}")
print(f"J_inv: {params.dynamics.J_inv}") 
print(f"motor_time_constant: {params.dynamics.motor_time_constant}")
print(f"integration dt: {params.integration.dt}")
# sample_initial_state(device, env, params, state, rng)
initial_state(device, env, params, state)
trajectory = []
N_STEPS = 5
for step_i in range(N_STEPS):
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


# plt.plot([s.position[2] for s in trajectory])
# plt.show()

vector_env = vector.VectorEnvironment()
vector_params = vector.VectorParameters()
vector_state = vector.VectorState()
vector_next_state = vector.VectorState()

vector.initialize_environment(device, vector_env)
vector_rng = vector.VectorRng()
vector.initialize_rng(device, vector_rng, 0)
vector.sample_initial_parameters(device, vector_env, vector_params, vector_rng)
vector.initial_state(device, vector_env, vector_params, vector_state)
vector_params.parameters = [copy.copy(params) for _ in range(vector_env.N_ENVIRONMENTS)]
vector_env_parameters_json = parameters_to_json(device, vector_env.environments[0], vector_params.parameters[0])
assert(vector_env_parameters_json == single_env_params_json)

vector_trajectory = []
N_STEPS = 100000
step_time = 0
assign_time = 0
start = time.time()
dtype = np.float32
action = np.ones((vector_env.N_ENVIRONMENTS, vector_env.ACTION_DIM), dtype=dtype)
rewards = np.ones((vector_env.N_ENVIRONMENTS), dtype=dtype)
truncated_flags = np.ones((vector_env.N_ENVIRONMENTS), dtype=np.bool)
for step_i in range(N_STEPS):
    vector.sample_initial_parameters_if_truncated(device, vector_env, vector_params, truncated_flags, vector_rng)
    vector.sample_initial_state_if_truncated(device, vector_env, vector_params, vector_state, truncated_flags, vector_rng)
    vector.step(device, vector_env, vector_params, vector_state, action, vector_next_state, vector_rng)
    vector.reward(device, vector_env, vector_params, vector_state, action, vector_next_state, rewards, vector_rng)
    vector.terminated(device, vector_env, vector_params, vector_state, truncated_flags, vector_rng)
    vector_state.assign(vector_next_state)





end = time.time()


print(f"Steps per second: {N_STEPS * vector_env.N_ENVIRONMENTS / (end - start)}")
print(f"Step share: {step_time / (end - start)}")   
print(f"Assign share: {assign_time / (end - start)}")