template <TI T_N_ENVIRONMENTS>
struct VectorEnvironment{
    static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
    std::array<ENVIRONMENT, N_ENVIRONMENTS> environments;
};

template <TI T_N_ENVIRONMENTS>
struct VectorParameters{
    static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
    std::array<ENVIRONMENT::Parameters, N_ENVIRONMENTS> parameters;
};

template <TI T_N_ENVIRONMENTS>
struct VectorState{
    static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
    std::array<ENVIRONMENT::State, N_ENVIRONMENTS> states;
};


template <TI N_ENVIRONMENTS>
void initialize_environment(DEVICE &device, VectorEnvironment<N_ENVIRONMENTS>& env){
    for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::malloc(device, env.environments[env_i]);
        rlt::init(device, env.environments[env_i]);
    }
}


void step(DEVICE& device, VectorEnvironment<N_ENVIRONMENTS>& env, VectorParameters<N_ENVIRONMENTS>& parameters, VectorState<N_ENVIRONMENTS>& states, py::array_t<T> actions, RNG& rng){
    if (actions.size() != N_ENVIRONMENTS * 4){
        throw std::runtime_error(std::format("Expected {} actions, got {}", N_ENVIRONMENTS * 4, actions.size()));
    }
    if(actions.ndim != 2){
        throw std::runtime_error(std::format("Expected 2D array (N_ENVIRONMENTS x ACTION_DIM), got {}D", actions.ndim));
    }
    if(actions.shape[0] != N_ENVIRONMENTS){
        throw std::runtime_error(std::format("Expected {} environments, got {}", N_ENVIRONMENTS, actions.shape[0]));
    }
    if(actions.shape[1] != 4){
        throw std::runtime_error(std::format("Expected 4 actions, got {}", actions.shape[1]));
    }
    if(actions.strides[0] != 4 * sizeof(T)){
        throw std::runtime_error(std::format("Expected contiguous array, got stride of {}", actions.strides[0]));
    }
    auto buf = action.request();
    T *data = static_cast<T*>(buf.ptr);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, true>> motor_commands = {data};
    for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
        auto view = rlt::row(device, motor_commands, env_i);
        step(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], actions[env_i], states.states[env_i], rng);
    }
}