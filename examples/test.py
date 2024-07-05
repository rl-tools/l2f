from l2f import *

device = Device()

rng = RNG()

env = Environment()
params = Parameters()
state = State()
next_state = State()
observation = Observation()
next_observation = Observation()
action = Action()
action.motor_command = [0, 0, 0, 0]
init(device, env, params)


sample_initial_parameters(device, env, params, rng);
sample_initial_state(device, env, params, state, rng);
step(device, env, params, state, action, next_state, rng);
observe(device, env, params, state, observation, rng);
observe(device, env, params, next_state, next_observation, rng);
