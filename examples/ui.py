from l2f import *
from l2f import vector1024 as vector
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy
import time
import asyncio
import websockets
import json


# test()

device = Device()

rng = Rng()

env = Environment()
ui = UI()
ui.ns = "l2f"
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



async def main():
    # uri = "ws://localhost:13337/backend"
    uri = "ws://localhost:8080/backend"
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        print(f"Handshake: {handshake}")
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        print(f"Namespace: {namespace}")
        ui.ns = namespace
        ui_message = set_ui_message(device, env, ui)
        parameters_message = set_parameters_message(device, env, params, ui)

        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        for step_i in range(100):
            # sleep for 1 second

            step(device, env, params, state, [1, 0, 0, 0], next_state, rng)
            state.assign(next_state)
            state_action_message = set_state_action_message(device, env, params, ui, state, [0, 0, 0, 0])
            await websocket.send(state_action_message)
            await asyncio.sleep(0.1)




if __name__ == "__main__":
    asyncio.run(main())