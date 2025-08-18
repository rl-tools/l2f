from l2f import *
from l2f import vector1024 as vector
import sys
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
params = Parameters()
state = State()
observation = Observation()
next_state = State()
observation = Observation()
initialize_environment(device, env)
initialize_rng(device, rng, 0)




sample_initial_parameters(device, env, params, rng)
params_json = parameters_to_json(device, env, params)
params_json = json.loads(params_json)
params_json["dynamics"]["mass"] *= 0.1
parameters_from_json(device, env, json.dumps(params_json), params)
print(f"rotor_positions: {params_json['dynamics']['rotor_positions']}")
print(f"rotor_thrust_coefficients: {params_json['dynamics']['rotor_thrust_coefficients']}")
print(f"J: {params_json['dynamics']['J']}")
print(f"J_inv: {params_json['dynamics']['J_inv']}") 
print(f"rotor_time_constants_rising: {params_json['dynamics']['rotor_time_constants_rising']}")
print(f"rotor_time_constants_falling: {params_json['dynamics']['rotor_time_constants_falling']}")
print(f"integration dt: {params_json['integration']['dt']}")
# sample_initial_state(device, env, params, state, rng)
initial_state(device, env, params, state)



async def main():
    uri = "ws://localhost:13337/backend"
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
            action = [1, 0, 0, 0]
            step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            state_action_message = set_state_action_message(device, env, params, ui, state, [0, 0, 0, 0])
            await websocket.send(state_action_message)
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())