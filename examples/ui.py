from l2f import *
from l2f import vector1024 as vector
import sys
import copy
import time
import asyncio
import websockets
import json
import base64
from foundation_policy import QuadrotorPolicy

policy = QuadrotorPolicy()

policy.reset()

# test()
USE_MESH = False
USE_MESH = True

device = Device()

rng = Rng()

env = Environment()
ui = UI()
params = Parameters()
state = State()
observation = Observation()
next_state = State()
initialize_environment(device, env)
initialize_rng(device, rng, 0)




sample_initial_parameters(device, env, params, rng)
params_json = parameters_to_json(device, env, params)
params_json = json.loads(params_json)
# params_json["dynamics"]["mass"] *= 0.1
parameters_from_json(device, env, json.dumps(params_json), params)
print(f"rotor_positions: {params_json['dynamics']['rotor_positions']}")
print(f"rotor_thrust_coefficients: {params_json['dynamics']['rotor_thrust_coefficients']}")
print(f"J: {params_json['dynamics']['J']}")
print(f"J_inv: {params_json['dynamics']['J_inv']}") 
print(f"rotor_time_constants_rising: {params_json['dynamics']['rotor_time_constants_rising']}")
print(f"rotor_time_constants_falling: {params_json['dynamics']['rotor_time_constants_falling']}")
print(f"integration dt: {params_json['integration']['dt']}")
sample_initial_state(device, env, params, state, rng)
# initial_state(device, env, params, state)




async def main():
    uri = "ws://localhost:13337/backend"
    # uri = "ws://localhost:8080/backend"
    async with websockets.connect(uri, max_size=100_000_000) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        print(f"Handshake: {handshake}")
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        print(f"Namespace: {namespace}")
        ui.ns = namespace
        ui_message = set_ui_message(device, env, ui)
        ui_message = json.loads(ui_message)
        # ui_message["data"]["debug"] = "ui.js"
        ui_message["data"]["options"] = {"showAxes": True}
        ui_message = json.dumps(ui_message)
        parameters_message = set_parameters_message(device, env, params, ui)
        parameters_message = json.loads(parameters_message)
        parameters_message["data"]["ui"] = {
            "model": "11f470c8206d4ca43bf3f7e1ba1d7acc456d3c34",
            "name": "x500"
        }
        parameters_message = json.dumps(parameters_message)
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        for step_i in range(1000):
            # sleep for 1 second

            l2f.observe(device, env, params, state, observation, rng)
            action = policy.evaluate_step(observation.observation[:3+9+3+3+4])
            dt = l2f.step(device, env, params, state, action.squeeze(), next_state, rng)
            state.assign(next_state)
            state_action_message = set_state_action_message(device, env, params, ui, state, [0, 0, 0, 0])
            await websocket.send(state_action_message)
            await asyncio.sleep(dt)

if __name__ == "__main__":
    asyncio.run(main())
