#include "common.h"

int test(){
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    RNG rng;
    initialize_environment(device, env);
    initialize_rng(device, rng, 0);

    ENVIRONMENT::State state, next_state;
    std::array<T, 4> action;
    std::array<T, ENVIRONMENT::OBSERVATION_DIM> observation, next_observation;
    sample_initial_parameters(device, env, parameters, rng);
    sample_initial_state(device, env, parameters, state, rng);
    for(TI action_i=0; action_i < 4; action_i++){
        action[action_i] = 0.0;
    }
    for(TI step_i=0; step_i < 100; step_i++){
        std::cout << "step: " << step_i << " position: " << state.position[0] << " " << state.position[1] << " " << state.position[2] << std::endl;
        step(device, env, parameters, state, action, next_state, rng);
        state = next_state;
    }
    // observe(device, env, parameters, state, observation, rng);
    // observe(device, env, parameters, next_state, next_observation, rng);
    // for(TI observation_i=0; observation_i < ENVIRONMENT::OBSERVATION_DIM; observation_i++){
    //     std::cout << observation.observation[observation_i] << " " << next_observation.observation[observation_i] << std::endl;
    // }
    std::string json = parameters_to_json(device, env, parameters);
    std::cout << json << std::endl;
    return 0;
}


int main(){
    test();
    return 0;
}