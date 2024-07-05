#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#ifdef RL_TOOLS_ENABLE_JSON
// for loading the config
#include <nlohmann/json.hpp>
#include <learning_to_fly/simulator/operations_cpu.h>
#endif

namespace rlt = rl_tools;

#ifndef TEST
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

namespace static_parameter_builder{
    // to prevent spamming the global namespace
    using namespace rl_tools::rl::environments::l2f;
    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        using STATE_BASE = StateBase<T, TI>;
        using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, StateRandomForce<T, TI, STATE_BASE>>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                        observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                        observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                        observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                        observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>
                                        >
                                        >
                                >>
                        >>
                >>
        >>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETER_FACTORY = parameters::DefaultParameters<T, TI>;
        static constexpr auto PARAMETER_VALUES = PARAMETER_FACTORY::parameters;
        using PARAMETERS = typename PARAMETER_FACTORY::PARAMETERS_TYPE;
    };
}

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, static_parameter_builder::ENVIRONMENT_STATIC_PARAMETERS>;
using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;

struct Device{
    DEVICE device;
};

struct Rng{
    RNG rng;
};

struct Environment{
    ENVIRONMENT env;
};

struct Parameters{
    ENVIRONMENT::Parameters parameters;
};

void initialize_rng(Device &device, Rng& rng, TI seed){
    rng.rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);
}

void initialize_environment(Device &device, Environment& env, Parameters& parameters){
    rlt::malloc(device.device, env.env);
    rlt::init(device.device, env.env, parameters.parameters);
}


struct State{
    std::array<T, 3> position;
    std::array<T, 4> orientation;
    std::array<T, 3> linear_velocity;
    std::array<T, 3> angular_velocity;
    std::array<T, 4> rpm;
    ENVIRONMENT::State state;
};

void sync(State& state, typename ENVIRONMENT::State& state_real){
    state.position[0] = state_real.position[0];
    state.position[1] = state_real.position[1];
    state.position[2] = state_real.position[2];
    state.orientation[0] = state_real.orientation[0];
    state.orientation[1] = state_real.orientation[1];
    state.orientation[2] = state_real.orientation[2];
    state.orientation[3] = state_real.orientation[3];
    state.linear_velocity[0] = state_real.linear_velocity[0];
    state.linear_velocity[1] = state_real.linear_velocity[1];
    state.linear_velocity[2] = state_real.linear_velocity[2];
    state.angular_velocity[0] = state_real.angular_velocity[0];
    state.angular_velocity[1] = state_real.angular_velocity[1];
    state.angular_velocity[2] = state_real.angular_velocity[2];
    state.rpm[0] = state_real.rpm[0];
    state.rpm[1] = state_real.rpm[1];
    state.rpm[2] = state_real.rpm[2];
    state.rpm[3] = state_real.rpm[3];
}
struct Action{
    std::array<T, ENVIRONMENT::ACTION_DIM> motor_command;
};

struct Observation{
    std::array<T, ENVIRONMENT::OBSERVATION_DIM> observation;
};

#ifdef RL_TOOLS_ENABLE_JSON
void load_config(Device& device, ENVIRONMENT& env, std::string config_string){
    nlohmann::json parameters_json = nlohmann::json::parse(config_string);
    rlt::load_config(device.device, env.parameters, parameters_json);
}
#endif

T step(Device& device, Environment& env, Parameters& parameters, State& state, Action action, State& next_state, Rng& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> motor_commands;
    for(TI action_i=0; action_i < 4; action_i++){
        set(motor_commands, 0, action_i, action.motor_command[action_i]);
    }
    T dt = rlt::step(device.device, env.env, parameters.parameters, state.state, motor_commands, next_state.state, rng.rng);
    sync(next_state, next_state.state);
    return dt;
}
void initial_parameters(Device& device, Environment& env, Parameters& parameters){
    rlt::initial_parameters(device.device, env.env, parameters.parameters);
}
void sample_initial_parameters(Device& device, Environment& env, Parameters& parameters, Rng& rng){
    rlt::sample_initial_parameters(device.device, env.env, parameters.parameters, rng.rng);
}
void initial_state(Device& device, Environment& env, Parameters& parameters, State& state){
    rlt::initial_state(device.device, env.env, parameters.parameters, state.state);
    sync(state, state.state);
}
void sample_initial_state(Device& device, Environment& env, Parameters& parameters, State& state, Rng& rng){
    rlt::sample_initial_state(device.device, env.env, parameters.parameters, state.state, rng.rng);
    sync(state, state.state);
}
void observe(Device& device, Environment& env, Parameters& parameters, State& state, Observation& observation, Rng& rng){
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation_matrix;
    rlt::observe(device.device, env.env, parameters.parameters, state.state, ENVIRONMENT::Observation{}, observation_matrix, rng.rng);
    for(TI observation_i=0; observation_i < ENVIRONMENT::OBSERVATION_DIM; observation_i++){
        observation.observation[observation_i] = get(observation_matrix, 0, observation_i);
    }
}

std::string parameters_to_json(Device& device, Environment& env, Parameters& parameters){
    return rlt::json(device.device, env.env, parameters.parameters);
}


int test(){
    Device device;
    Environment env;
    Parameters parameters;
    Rng rng;
    initialize_environment(device, env, parameters);
    initialize_rng(device, rng, 0);

    State state, next_state;
    Action action;
    Observation observation, next_observation;
    sample_initial_parameters(device, env, parameters, rng);
    sample_initial_state(device, env, parameters, state, rng);
    for(TI action_i=0; action_i < 4; action_i++){
        action.motor_command[action_i] = 0.0;
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

class Calculator {
public:
    Calculator() = default;
    int add(int a, int b) { return a + b; }
    int subtract(int a, int b) { return a - b; }
};


#ifndef TEST
PYBIND11_MODULE(interface, m) {
    // Optional: m.doc() = "Documentation string for the module"; // Module documentation
    py::class_<Device>(m, "Device")
        .def(py::init<>());
    py::class_<Rng>(m, "Rng")
        .def(py::init<>());
    py::class_<Environment>(m, "Environment")
        .def(py::init<>());
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>());
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def_readwrite("position", &State::position)
        .def_readwrite("orientation", &State::orientation)
        .def_readwrite("linear_velocity", &State::linear_velocity)
        .def_readwrite("angular_velocity", &State::angular_velocity)
        .def_readwrite("rpm", &State::rpm);
    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("motor_command", &Action::motor_command);
    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("observation", &Observation::observation);


    m.def("initialize_environment", &initialize_environment, "Init environement");
    m.def("initialize_rng", &initialize_rng, "Init Rng");
    m.def("step", &step, "Simulate one step");
    m.def("initial_parameters", &initial_parameters, "Reset to default parameters");
    m.def("sample_initial_parameters", &sample_initial_parameters, "Reset to random parameters");
    m.def("initial_state", &initial_state, "Reset to default state");
    m.def("sample_initial_state", &sample_initial_state, "Reset to random state");
    m.def("observe", &observe, "Observe state");
    m.def("parameters_to_json", &parameters_to_json, "Convert parameters to json");
    m.def("test", &test, "Test");
#ifdef RL_TOOLS_ENABLE_JSON
    m.def("load_config", &load_config, "Load config");
#endif
}
#else
int main(){
    test();
    return 0;
}

#endif