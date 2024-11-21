#include <sstream>
#include <stdexcept>
#include <pybind11/numpy.h>


namespace vector{
    template <TI T_N_ENVIRONMENTS>
    struct Environment{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        std::array<ENVIRONMENT, N_ENVIRONMENTS> environments;
    };

    template <TI T_N_ENVIRONMENTS>
    struct Parameters{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        std::array<ENVIRONMENT::Parameters, N_ENVIRONMENTS> parameters;
    };

    template <TI T_N_ENVIRONMENTS>
    struct State{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        std::array<ENVIRONMENT::State, N_ENVIRONMENTS> states;
    };


    template <TI N_ENVIRONMENTS>
    void initialize_environment(DEVICE &device, Environment<N_ENVIRONMENTS>& env){
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::malloc(device, env.environments[env_i]);
            rlt::init(device, env.environments[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void initial_parameters(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters){
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::initial_parameters(device, env.environments[env_i], parameters.parameters[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void sample_initial_parameters(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, RNG& rng){
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::sample_initial_parameters(device, env.environments[env_i], parameters.parameters[env_i], rng);
        }
    }

    template <TI N_ENVIRONMENTS>
    void initial_state(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states){
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::initial_state(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void sample_initial_state(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, RNG& rng){
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::sample_initial_state(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], rng);
        }
    }

    template <TI N_ENVIRONMENTS>
    void step(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array actions, State<N_ENVIRONMENTS>& next_states, RNG& rng){

        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Expected float or double array");
        if(!actions.dtype().is(py::dtype::of<T>())){
            std::ostringstream oss;
            std::string expected_type = std::is_same_v<T, float> ? "float" : "double";
            oss << "Expected " << expected_type << " array, got " << actions.dtype().str();
            throw std::runtime_error(oss.str());
        }

        if (actions.size() != N_ENVIRONMENTS * 4) {
            std::ostringstream oss;
            oss << "Expected " << (N_ENVIRONMENTS * 4) << " actions, got " << actions.size();
            throw std::runtime_error(oss.str());
        }

        if (actions.ndim() != 2) {
            std::ostringstream oss;
            oss << "Expected 2D array (N_ENVIRONMENTS x ACTION_DIM), got " << actions.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto shape = actions.shape();
        if (shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS << " environments, got " << shape[0];
            throw std::runtime_error(oss.str());
        }

        if (shape[1] != 4) {
            std::ostringstream oss;
            oss << "Expected 4 actions, got " << shape[1];
            throw std::runtime_error(oss.str());
        }

        if (actions.strides()[0] != 4 * sizeof(T)) {
            std::ostringstream oss;
            oss << "Expected stride " << 4 * sizeof(T) << ", got stride of " << actions.strides()[0];
            throw std::runtime_error(oss.str());
        }

        auto buf = actions.request();
        T *data = static_cast<T*>(buf.ptr);
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, true>> motor_commands = {data};
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            auto view = rlt::row(device, motor_commands, env_i);
            rlt::step(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], view, next_states.states[env_i], rng);
        }
    }

    template <TI N_ENVIRONMENTS>
    void observe(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array observations, RNG& rng){
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Expected float or double array");
        if(!observations.dtype().is(py::dtype::of<T>())){
            std::ostringstream oss;
            std::string expected_type = std::is_same_v<T, float> ? "float" : "double";
            oss << "Expected " << expected_type << " array, got " << observations.dtype().str();
            throw std::runtime_error(oss.str());
        }
        if (observations.size() != N_ENVIRONMENTS * ENVIRONMENT::Observation::DIM) {
            std::ostringstream oss;
            oss << "Expected " << (N_ENVIRONMENTS * ENVIRONMENT::Observation::DIM)
                << " observations, got " << observations.size();
            throw std::runtime_error(oss.str());
        }

        if (observations.ndim() != 2) {
            std::ostringstream oss;
            oss << "Expected 2D array (N_ENVIRONMENTS x OBSERVATION_DIM), got " 
                << observations.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto shape = observations.shape();
        if (shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS 
                << " environments, got " << shape[0];
            throw std::runtime_error(oss.str());
        }

        if (shape[1] != ENVIRONMENT::Observation::DIM) {
            std::ostringstream oss;
            oss << "Expected " << ENVIRONMENT::Observation::DIM 
                << " observations, got " << shape[1];
            throw std::runtime_error(oss.str());
        }

        if (observations.strides()[0] != ENVIRONMENT::Observation::DIM * sizeof(T)) {
            std::ostringstream oss;
            oss << "Expected contiguous array, got stride of " 
                << observations.strides()[0];
            throw std::runtime_error(oss.str());
        }

        auto buf = observations.request();
        T *data = static_cast<T*>(buf.ptr);
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM, true>> observation_matrix = {data};
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            auto view = rlt::row(device, observation_matrix, env_i);
            rlt::observe(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], ENVIRONMENT::Observation{}, view, rng);
        }
    }

}
