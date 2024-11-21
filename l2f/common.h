#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>


#include <array>

namespace rlt = rl_tools;


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
        using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, false, StateRandomForce<T, TI, STATE_BASE>>;
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

void initialize_rng(DEVICE &device, RNG& rng, TI seed){
    rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);
}

void initialize_environment(DEVICE &device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters){
    rlt::malloc(device, env);
    rlt::init(device, env);
}

T step(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters, ENVIRONMENT::State& state, std::array<T, 4> action, ENVIRONMENT::State& next_state, RNG& rng){
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> motor_commands;
    for(TI action_i=0; action_i < 4; action_i++){
        set(motor_commands, 0, action_i, action[action_i]);
    }
    T dt = rlt::step(device, env, parameters, state, motor_commands, next_state, rng);
    return dt;
}
void initial_parameters(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters){
    rlt::initial_parameters(device, env, parameters);
}
void sample_initial_parameters(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters, RNG& rng){
    rlt::sample_initial_parameters(device, env, parameters, rng);
}
void initial_state(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters, ENVIRONMENT::State& state){
    rlt::initial_state(device, env, parameters, state);
}
void sample_initial_state(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters, ENVIRONMENT::State& state, RNG& rng){
    rlt::sample_initial_state(device, env, parameters, state, rng);
}
void observe(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters, ENVIRONMENT::State& state, std::array<T, ENVIRONMENT::Observation::DIM>& observation, RNG& rng){
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM, false>> observation_matrix;
    rlt::observe(device, env, parameters, state, ENVIRONMENT::Observation{}, observation_matrix, rng);
    for(TI observation_i=0; observation_i < ENVIRONMENT::OBSERVATION_DIM; observation_i++){
        observation[observation_i] = get(observation_matrix, 0, observation_i);
    }
}

std::string parameters_to_json(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::Parameters& parameters){
    return rlt::json(device, env, parameters);
}
