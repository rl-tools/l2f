# L2F: Learning to Fly Simulator


```
pip install l2f
```

This repo contains Python bindings for the simulator introduced in [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081).

Please check the [example](./examples/test.py) for how to use it.

## CUDA

For the CUDA usage please refer to [rl-tools/l2f-benchmark](https://github.com/rl-tools/l2f-benchmark)


## Getting Started

Install UIServer (forwards states to HTML/JS UI) and the foundation-policy (a general quadrotor policy)

```
pip install l2f ui-server foundation-policy
```


Run the UIServer
```
ui-server
```
Navigate to [http://localhost:13337](http://localhost:13337)
This is run separately from the client code such that you can keep the browser open, without windows popping up and going away when re-running your code. Also the camera perspective is maintained across runs.

Then run e.g.:

```python
from copy import copy
import numpy as np
import asyncio, websockets, json
import l2f
from l2f import vector8 as vector
from foundation_policy import QuadrotorPolicy

policy = QuadrotorPolicy()
device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
ui = l2f.UI()
params = vector.VectorParameters()
state = vector.VectorState()
observation = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)
next_state = vector.VectorState()

vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
vector.sample_initial_state(device, env, params, state, rng)

def configure_3d_model(parameters_message):
    parameters_message = json.loads(parameters_message)
    for d in parameters_message["data"]:
        d["ui"] = {
            "model": "95d22881d444145176db6027d44ebd3a15e9699a",
            "name": "x500"
        }
    return json.dumps(parameters_message)

async def render(websocket, state, action):
    ui_state = copy(state)
    for i, s in enumerate(ui_state.states):
        s.position[0] += i * 0.1 # Spacing for visualization
    state_action_message = vector.set_state_action_message(device, env, params, ui, ui_state, action)
    await websocket.send(state_action_message)

async def main():
    uri = "ws://localhost:13337/backend" # connection to the UI server
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = vector.set_ui_message(device, env, ui)
        parameters_message = vector.set_parameters_message(device, env, params, ui)
        # parameters_message = configure_3d_model(parameters_message) # use this for a more realistic 3d model
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        await asyncio.sleep(1)
        await render(websocket, state, np.zeros((8, 4)))
        await asyncio.sleep(2)
        policy.reset()
        for _ in range(500):
            vector.observe(device, env, params, state, observation, rng)
            action = policy.evaluate_step(observation[:, :22])
            dts = vector.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            await render(websocket, state, action)
            await asyncio.sleep(dts[-1])

if __name__ == "__main__":
    asyncio.run(main())

```
