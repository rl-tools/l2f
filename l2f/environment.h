#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>

namespace env_builder{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI>
    struct ENVIRONMENT_FACTORY{

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::Squared<T>;
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                00.10, // scale
                01.00, // constant
                00.00, // termination penalty
                10.00, // position
                02.50, // orientation
                00.05, // linear_velocity
                00.00, // angular_velocity
                00.00, // linear_acceleration
                00.00, // angular_acceleration
                00.10, // action
        };

        struct TRAJECTORY_OPTIONS{
            static constexpr bool LANGEVIN = false;
        };
        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool ENABLED = false;
            static constexpr bool THRUST_TO_WEIGHT = ENABLED;
            static constexpr bool MASS = ENABLED;
            static constexpr bool TORQUE_TO_INERTIA = ENABLED;
            static constexpr bool MASS_SIZE_DEVIATION = ENABLED;
            static constexpr bool ROTOR_TORQUE_CONSTANT = ENABLED;
            static constexpr bool DISTURBANCE_FORCE = ENABLED;
            static constexpr bool ROTOR_TIME_CONSTANT = ENABLED;
        };
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, TRAJECTORY_OPTIONS, ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>>>;

        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        static constexpr TI SIMULATION_FREQUENCY = 100;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = rl_tools::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
            0, // position
            0, // orientation
            0, // linear_velocity
            0, // angular_velocity
            0, // imu acceleration
        };
        static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
            0, // std of additive gaussian noise onto the normalized action (-1, 1)
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
            true,  // enable
            1,     // position
            10,    // linear velocity
            35,    // angular velocity
            10000, // position integral
            50000, // orientation integral
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            observation_noise,
            action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
            0.0, // thrust_to_weight_min;
            0.0, // thrust_to_weight_max;
            0.0, // thrust_to_weight_by_torque_to_inertia_min;
            0.0, // thrust_to_weight_by_torque_to_inertia_max;
            0.0, // mass_min;
            0.0, // mass_max;
            0.0, // mass_size_deviation;
            0.0, // motor_time_constant;
            0.0  // rotor_torque_constant;
        };
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, // random_force;
            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
        };
        static constexpr typename PARAMETERS_TYPE::Trajectory trajectory = {
            {1.0, 0.0}, // mixture weights
            typename PARAMETERS_TYPE::Trajectory::Langevin{
                1.00, // gamma
                2.00, // omega
                0.50, // sigma
                0.01 // alpha
            }
        };
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        dynamics,
                        integration,
                        mdp
                    }, // Base
                    disturbances
                }, // Disturbances
                domain_randomization
            }, // DomainRandomization
            trajectory // Trajectory
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 16;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; 
            static constexpr TI ANGULAR_VELOCITY_HISTORY = ANGULAR_VELOCITY_DELAY;
            using STATE_TYPE = StateTrajectory<StateSpecification<T, TI, DefaultActionHistoryState<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_HISTORY>>>;
            using OBSERVATION_TYPE = DefaultActionHistoryObservation<T, TI, ACTION_HISTORY_LENGTH, ANGULAR_VELOCITY_DELAY>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}

