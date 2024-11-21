#include "common.h"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(interface, m) {
    // Optional: m.doc() = "Documentation string for the module"; // Module documentation
    py::class_<DEVICE>(m, "Device")
        .def(py::init<>());
    py::class_<RNG>(m, "Rng")
        .def(py::init<>());
    py::class_<ENVIRONMENT>(m, "Environment")
        .def(py::init<>());
    py::class_<ENVIRONMENT::Parameters>(m, "Parameters")
        .def(py::init<>())
        .def("__copy__", [](const ENVIRONMENT::Parameters &self) {
            return ENVIRONMENT::Parameters(self);
        })
        .def_readwrite("dynamics", &ENVIRONMENT::Parameters::dynamics)
        .def_readwrite("integration", &ENVIRONMENT::Parameters::integration);
    py::class_<ENVIRONMENT::Parameters::Dynamics>(m, "DynamicsParameters")
        .def(py::init<>())
        .def("__copy__", [](const ENVIRONMENT::Parameters::Dynamics &self) {
            return ENVIRONMENT::Parameters::Dynamics(self);
        })
        .def_property("rotor_positions",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> {
                std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> result;
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(std::begin(self.rotor_positions[i]), std::end(self.rotor_positions[i]), result[i].begin());
                }
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> &new_data) {
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(new_data[i].begin(), new_data[i].end(), std::begin(self.rotor_positions[i]));
                }
            }
        )
        .def_property("rotor_thrust_directions",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> {
                std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> result;
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(std::begin(self.rotor_thrust_directions[i]), std::end(self.rotor_thrust_directions[i]), result[i].begin());
                }
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> &new_data) {
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(new_data[i].begin(), new_data[i].end(), std::begin(self.rotor_thrust_directions[i]));
                }
            }
        )
        .def_property("rotor_torque_directions",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> {
                std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> result;
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(std::begin(self.rotor_torque_directions[i]), std::end(self.rotor_torque_directions[i]), result[i].begin());
                }
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<std::array<T, 3>, ENVIRONMENT::Parameters::N> &new_data) {
                for (TI i = 0; i < ENVIRONMENT::Parameters::N; ++i) {
                    std::copy(new_data[i].begin(), new_data[i].end(), std::begin(self.rotor_torque_directions[i]));
                }
            }
        )
        .def_property("rotor_thrust_coefficients",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<T, 3> {
                std::array<T, 3> result;
                std::copy(std::begin(self.rotor_thrust_coefficients), std::end(self.rotor_thrust_coefficients), result.begin());
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<T, 3> &new_data) {
                std::copy(new_data.begin(), new_data.end(), std::begin(self.rotor_thrust_coefficients));
            }
        )
        .def_readwrite("rotor_torque_constant", &ENVIRONMENT::Parameters::Dynamics::rotor_torque_constant)
        .def_readwrite("mass", &ENVIRONMENT::Parameters::Dynamics::mass)
        .def_property("J",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<std::array<T, 3>, 3> {
                std::array<std::array<T, 3>, 3> result;
                for (TI i = 0; i < 3; ++i) {
                    std::copy(std::begin(self.J[i]), std::end(self.J[i]), result[i].begin());
                }
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<std::array<T, 3>, 3> &new_data) {
                for (TI i = 0; i < 3; ++i) {
                    std::copy(new_data[i].begin(), new_data[i].end(), std::begin(self.J[i]));
                }
            }
        )
        .def_property("J_inv",
            [](ENVIRONMENT::Parameters::Dynamics &self) -> std::array<std::array<T, 3>, 3> {
                std::array<std::array<T, 3>, 3> result;
                for (TI i = 0; i < 3; ++i) {
                    std::copy(std::begin(self.J_inv[i]), std::end(self.J_inv[i]), result[i].begin());
                }
                return result;
            },
            [](ENVIRONMENT::Parameters::Dynamics &self, const std::array<std::array<T, 3>, 3> &new_data) {
                for (TI i = 0; i < 3; ++i) {
                    std::copy(new_data[i].begin(), new_data[i].end(), std::begin(self.J_inv[i]));
                }
            }
        )
        .def_readwrite("motor_time_constant", &ENVIRONMENT::Parameters::Dynamics::motor_time_constant);
    py::class_<ENVIRONMENT::Parameters::Integration>(m, "IntegrationParameters")
        .def(py::init<>())
        .def_readwrite("dt", &ENVIRONMENT::Parameters::Integration::dt);
    py::class_<ENVIRONMENT::State>(m, "State")
    .def(py::init<>())
    .def("__copy__", [](const ENVIRONMENT::State &self) {
        return ENVIRONMENT::State(self);
    })
    .def("assign", [](ENVIRONMENT::State &self, const ENVIRONMENT::State &other) {
        self = other;
    })
    .def_property("position",
        [](ENVIRONMENT::State &self) -> std::array<T, 3> {
            std::array<T, 3> position;
            std::copy(std::begin(self.position), std::end(self.position), position.begin());
            return position;
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.position));
        }
    )
    .def_property("orientation",
        [](ENVIRONMENT::State &self) -> std::array<T, 4> {
            std::array<T, 4> orientation;
            std::copy(std::begin(self.orientation), std::end(self.orientation), orientation.begin());
            return orientation;
        },
        [](ENVIRONMENT::State &self, const std::array<T, 4> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.orientation));
        }
    )
    .def_property("linear_velocity",
        [](ENVIRONMENT::State &self) -> std::array<T, 3> {
            std::array<T, 3> linear_velocity;
            std::copy(std::begin(self.linear_velocity), std::end(self.linear_velocity), linear_velocity.begin());
            return linear_velocity;
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.linear_velocity));
        }
    )
    .def_property("angular_velocity",
        [](ENVIRONMENT::State &self) -> std::array<T, 3> {
            std::array<T, 3> angular_velocity;
            std::copy(std::begin(self.angular_velocity), std::end(self.angular_velocity), angular_velocity.begin());
            return angular_velocity;
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.angular_velocity));
        }
    )
    .def_property("rpm", 
        [](ENVIRONMENT::State &self) -> std::array<T, 4> {
            std::array<T, 4> rpm;
            std::copy(std::begin(self.rpm), std::end(self.rpm), rpm.begin());
            return rpm;
        },
        [](ENVIRONMENT::State &self, const std::array<T, 4> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.rpm));
        }
    );




    m.def("initialize_environment", &initialize_environment, "Init environement");
    m.def("initialize_rng", &initialize_rng, "Init Rng");
    m.def("step", &step, "Simulate one step");
    m.def("initial_parameters", &initial_parameters, "Reset to default parameters");
    m.def("sample_initial_parameters", &sample_initial_parameters, "Reset to random parameters");
    m.def("initial_state", &initial_state, "Reset to default state");
    m.def("sample_initial_state", &sample_initial_state, "Reset to random state");
    m.def("observe", &observe, "Observe state");
    m.def("parameters_to_json", &parameters_to_json, "Convert parameters to json");
}