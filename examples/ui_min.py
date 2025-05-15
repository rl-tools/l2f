import l2f
import asyncio, websockets, json
device = l2f.Device()
rng = l2f.Rng()
env = l2f.Environment()
ui = l2f.UI()
params = l2f.Parameters()
state = l2f.State()
next_state = l2f.State()
l2f.initialize_environment(device, env)
l2f.initialize_rng(device, rng, 0)
l2f.sample_initial_parameters(device, env, params, rng)
l2f.initial_state(device, env, params, state)
async def main():
    uri = "ws://localhost:13337/backend"
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = l2f.set_ui_message(device, env, ui)
        parameters_message = l2f.set_parameters_message(device, env, params, ui)
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        for _ in range(100):
            action = [1, 0, 0, 0]
            dt = l2f.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            state_action_message = l2f.set_state_action_message(device, env, params, ui, state, action)
            await websocket.send(state_action_message)
            await asyncio.sleep(dt)
if __name__ == "__main__":
    asyncio.run(main())