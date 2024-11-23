#include <sstream>
#include <stdexcept>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef L2F_CACHE_LINE_SIZE
#define L2F_CACHE_LINE_SIZE 64
#endif

namespace vector{
    template <TI T_N_ENVIRONMENTS>
    struct Rng{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        alignas(L2F_CACHE_LINE_SIZE) std::array<RNG, N_ENVIRONMENTS> rngs;
    };
    template <TI T_N_ENVIRONMENTS>
    struct Environment{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        alignas(L2F_CACHE_LINE_SIZE) std::array<ENVIRONMENT, N_ENVIRONMENTS> environments;
    };

    template <TI T_N_ENVIRONMENTS>
    struct Parameters{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        alignas(L2F_CACHE_LINE_SIZE) std::array<ENVIRONMENT::Parameters, N_ENVIRONMENTS> parameters;
    };


    template <TI T_N_ENVIRONMENTS>
    struct State{
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        alignas(L2F_CACHE_LINE_SIZE) std::array<ENVIRONMENT::State, N_ENVIRONMENTS> states;
    };
    template <TI N_ENVIRONMENTS>
    void initialize_rng(DEVICE &device, Rng<N_ENVIRONMENTS>& rng, TI seed){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rng.rngs[env_i] = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed + env_i);
        }
    }

    template <TI N_ENVIRONMENTS>
    void initialize_environment(DEVICE &device, Environment<N_ENVIRONMENTS>& env){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::malloc(device, env.environments[env_i]);
            rlt::init(device, env.environments[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void initial_parameters(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::initial_parameters(device, env.environments[env_i], parameters.parameters[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void sample_initial_parameters(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, Rng<N_ENVIRONMENTS>& rng){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::sample_initial_parameters(device, env.environments[env_i], parameters.parameters[env_i], rng.rngs[env_i]);
        }
    }
    template <TI N_ENVIRONMENTS>
    void sample_initial_parameters_if_truncated(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, py::array_t<bool> truncated, Rng<N_ENVIRONMENTS>& rng){
        if(!truncated.dtype().is(py::dtype::of<bool>())){
            throw std::runtime_error("Expected bool array");
        }
        if (truncated.size() != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << (N_ENVIRONMENTS)
                << " truncated, got " << truncated.size();
            throw std::runtime_error(oss.str());
        }

        if (truncated.ndim() != 1) {
            std::ostringstream oss;
            oss << "Expected 1D array (N_ENVIRONMENTS,), got " 
                << truncated.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto shape = truncated.shape();
        if (shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS 
                << " environments, got " << shape[0];
            throw std::runtime_error(oss.str());
        }

        auto mutable_truncated = truncated.unchecked<1>();
        #pragma omp parallel for schedule(dynamic, 1)
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            if(mutable_truncated(env_i)){
                rlt::sample_initial_parameters(device, env.environments[env_i], parameters.parameters[env_i], rng.rngs[env_i]);
            }
        }
    }


    template <TI N_ENVIRONMENTS>
    void initial_state(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::initial_state(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void sample_initial_state(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, Rng<N_ENVIRONMENTS>& rng){
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::sample_initial_state(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], rng.rngs[env_i]);
        }
    }
    template <TI N_ENVIRONMENTS>
    void sample_initial_state_if_truncated(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array_t<bool> truncated, Rng<N_ENVIRONMENTS>& rng){
        if(!truncated.dtype().is(py::dtype::of<bool>())){
            throw std::runtime_error("Expected bool array");
        }
        if (truncated.size() != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << (N_ENVIRONMENTS)
                << " truncated, got " << truncated.size();
            throw std::runtime_error(oss.str());
        }

        if (truncated.ndim() != 1) {
            std::ostringstream oss;
            oss << "Expected 1D array (N_ENVIRONMENTS,), got " 
                << truncated.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto shape = truncated.shape();
        if (shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS 
                << " environments, got " << shape[0];
            throw std::runtime_error(oss.str());
        }

        auto mutable_truncated = truncated.unchecked<1>();
        #pragma omp parallel for schedule(dynamic, 1)
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            if(mutable_truncated(env_i)){
                rlt::sample_initial_state(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], rng.rngs[env_i]);
            }
        }
    }

    template <TI N_ENVIRONMENTS>
    void step(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array actions, State<N_ENVIRONMENTS>& next_states, Rng<N_ENVIRONMENTS>& rng){
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

        constexpr TI ACTION_SIZE = 4 * sizeof(T);
        if (actions.strides()[0] != ACTION_SIZE){
            std::ostringstream oss;
            oss << "Expected stride " << ACTION_SIZE << ", got stride of " << actions.strides()[0];
            throw std::runtime_error(oss.str());
        }

        auto buf = actions.request();
        T *data = static_cast<T*>(buf.ptr);
        rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ENVIRONMENT::ACTION_DIM, true>> motor_commands = {data};
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            auto view = rlt::row(device, motor_commands, env_i);
            rlt::step(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], view, next_states.states[env_i], rng.rngs[env_i]);
        }
    }

    template <TI N_ENVIRONMENTS>
    void observe(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array observations, Rng<N_ENVIRONMENTS>& rng){
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
        rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ENVIRONMENT::Observation::DIM, true>> observation_matrix = {data};
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            auto view = rlt::row(device, observation_matrix, env_i);
            rlt::observe(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], ENVIRONMENT::Observation{}, view, rng.rngs[env_i]);
        }
    }
    template <TI N_ENVIRONMENTS>
    void reward(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array actions, State<N_ENVIRONMENTS>& next_states, py::array rewards, Rng<N_ENVIRONMENTS>& rng){
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

        constexpr TI ACTION_SIZE = 4 * sizeof(T);
        if (actions.strides()[0] != ACTION_SIZE){
            std::ostringstream oss;
            oss << "Expected stride " << ACTION_SIZE << ", got stride of " << actions.strides()[0];
            throw std::runtime_error(oss.str());
        }
        if(!rewards.dtype().is(py::dtype::of<T>())){
            std::ostringstream oss;
            std::string expected_type = std::is_same_v<T, float> ? "float" : "double";
            oss << "Expected " << expected_type << " array, got " << rewards.dtype().str();
            throw std::runtime_error(oss.str());
        }
        if (rewards.size() != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS
                << " rewards, got " << rewards.size();
            throw std::runtime_error(oss.str());
        }

        if (rewards.ndim() != 1) {
            std::ostringstream oss;
            oss << "Expected 1D array (N_ENVIRONMENTS,), got " 
                << rewards.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto rewards_shape = rewards.shape();
        if (rewards_shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS 
                << " environments, got " << rewards_shape[0];
            throw std::runtime_error(oss.str());
        }

        auto actions_buf = actions.request();
        T *actions_data = static_cast<T*>(actions_buf.ptr);
        rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ENVIRONMENT::ACTION_DIM, true>> action_matrix = {actions_data};
        auto rewards_buf = rewards.request();
        T *rewards_data = static_cast<T*>(rewards_buf.ptr);
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, N_ENVIRONMENTS, true>> reward_matrix = {rewards_data};
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            auto view = rlt::row(device, action_matrix, env_i);
            T reward = rlt::reward(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], view, next_states.states[env_i], rng.rngs[env_i]);
            set(reward_matrix, 0, env_i, reward);
        }
    }
    template <TI N_ENVIRONMENTS>
    void terminated(DEVICE& device, Environment<N_ENVIRONMENTS>& env, Parameters<N_ENVIRONMENTS>& parameters, State<N_ENVIRONMENTS>& states, py::array_t<bool> terminated_flags, Rng<N_ENVIRONMENTS>& rng){
        if(!terminated_flags.dtype().is(py::dtype::of<bool>())){
            throw std::runtime_error("Expected bool array");
        }
        if (terminated_flags.size() != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << (N_ENVIRONMENTS)
                << " terminated_flags, got " << terminated_flags.size();
            throw std::runtime_error(oss.str());
        }

        if (terminated_flags.ndim() != 1) {
            std::ostringstream oss;
            oss << "Expected 1D array (N_ENVIRONMENTS,), got " 
                << terminated_flags.ndim() << "D";
            throw std::runtime_error(oss.str());
        }

        auto shape = terminated_flags.shape();
        if (shape[0] != N_ENVIRONMENTS) {
            std::ostringstream oss;
            oss << "Expected " << N_ENVIRONMENTS 
                << " environments, got " << shape[0];
            throw std::runtime_error(oss.str());
        }

        auto mutable_terminated_flags = terminated_flags.mutable_unchecked<1>();
        #pragma omp parallel for
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            bool terminated_flag_env = rlt::terminated(device, env.environments[env_i], parameters.parameters[env_i], states.states[env_i], rng.rngs[env_i]);
            mutable_terminated_flags(env_i) = terminated_flag_env;
        }
    }

}

