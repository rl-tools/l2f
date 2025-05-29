#include "common.h"


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#define L2F_VECTOR
// DEBUG: only compile vector8 to speed up compilation
// #define DEBUG
#ifdef L2F_VECTOR

#include <pybind11/numpy.h>
using DYNAMIC_ARRAY = py::array_t<T, py::array::c_style | py::array::forcecast>;

#include "vector.h"
#endif



template <TI N_ENVIRONMENTS>
py::module_ vector_factory(py::module_ &m){
    std::string name = "vector";
    name += std::to_string(N_ENVIRONMENTS);
    py::module_ vector = m.def_submodule(name.c_str());
    vector.def("initialize_environment", &vector::initialize_environment<N_ENVIRONMENTS>, "Init environement");
    vector.def("step", &vector::step<N_ENVIRONMENTS>, "Simulate one step");
    py::class_<vector::Environment<N_ENVIRONMENTS>>(vector, "VectorEnvironment")
        .def(py::init<>())
        .def_property_readonly("OBSERVATION_DIM", [](const vector::Environment<N_ENVIRONMENTS> &self) { return ENVIRONMENT::Observation::DIM; })
        .def_property_readonly("N_ENVIRONMENTS", [](const vector::Environment<N_ENVIRONMENTS> &self) { return vector::Environment<N_ENVIRONMENTS>::N_ENVIRONMENTS; })
        .def_property_readonly("ACTION_DIM", [](const vector::Environment<N_ENVIRONMENTS> &self) { return ENVIRONMENT::ACTION_DIM; })
        .def_property_readonly("EPISODE_STEP_LIMIT", [](const vector::Environment<N_ENVIRONMENTS> &self) { return ENVIRONMENT::EPISODE_STEP_LIMIT; })
        .def_readwrite("environments", &vector::Environment<N_ENVIRONMENTS>::environments);
    py::class_<vector::Parameters<N_ENVIRONMENTS>>(vector, "VectorParameters")
        .def(py::init<>())
        .def_readwrite("parameters", &vector::Parameters<N_ENVIRONMENTS>::parameters);
    py::class_<vector::State<N_ENVIRONMENTS>>(vector, "VectorState")
        .def(py::init<>())
        .def("assign", [](vector::State<N_ENVIRONMENTS> &self, const vector::State<N_ENVIRONMENTS> &other) {
            self = other;
        })
        .def("__copy__", [](const vector::State<N_ENVIRONMENTS> &self) {
            return vector::State<N_ENVIRONMENTS>(self);
        })
        .def_readwrite("states", &vector::State<N_ENVIRONMENTS>::states);
    py::class_<vector::Rng<N_ENVIRONMENTS>>(vector, "VectorRng")
        .def(py::init<>())
        .def_readwrite("rngs", &vector::Rng<N_ENVIRONMENTS>::rngs);
    vector.def("initialize_rng", &vector::initialize_rng<N_ENVIRONMENTS>, "Init rng");

    vector.def("initial_parameters", &vector::initial_parameters<N_ENVIRONMENTS>, "Reset to default parameters");
    vector.def("sample_initial_parameters", &vector::sample_initial_parameters<N_ENVIRONMENTS>, "Reset to random parameters");
    vector.def("sample_initial_parameters_if_truncated", &vector::sample_initial_parameters_if_truncated<N_ENVIRONMENTS>, "Reset to random parameters");
    vector.def("initial_state", &vector::initial_state<N_ENVIRONMENTS>, "Reset to default state");
    vector.def("sample_initial_state", &vector::sample_initial_state<N_ENVIRONMENTS>, "Reset to random state");
    vector.def("sample_initial_state_if_truncated", &vector::sample_initial_state_if_truncated<N_ENVIRONMENTS>, "Reset to random state");
    vector.def("step", &vector::step<N_ENVIRONMENTS>, "Simulate one step");
    vector.def("observe", &vector::observe<N_ENVIRONMENTS>, "Observe state");
    vector.def("reward", &vector::reward<N_ENVIRONMENTS>, "Get reward");
    vector.def("terminated", &vector::terminated<N_ENVIRONMENTS>, "Check if terminated");
    vector.def("set_parameters_message", &vector::set_parameters_message<N_ENVIRONMENTS>, "Set parameters message");
    vector.def("set_ui_message", &vector::set_ui_message<N_ENVIRONMENTS>, "Set ui message");
    vector.def("set_state_action_message", &vector::set_state_action_message<N_ENVIRONMENTS>, "Set state action message");
    return vector;
}


PYBIND11_MODULE(interface, m) {
    #ifdef DEBUG
    m.attr("DEBUG") = true;
    #else
    m.attr("DEBUG") = false;
    #endif
    py::class_<DEVICE>(m, "Device")
        .def(py::init<>());
    py::class_<RNG>(m, "Rng")
        .def(py::init<>());
    py::class_<ENVIRONMENT>(m, "Environment")
        .def(py::init<>())
        .def_property_readonly("OBSERVATION_DIM", [](const ENVIRONMENT &self) { return ENVIRONMENT::Observation::DIM; });
    py::class_<UI>(m, "UI")
        .def(py::init<>())
        .def_readwrite("ns", &UI::ns);
    m.def("set_parameters_message", &set_parameters_message, "Set parameters message");
    m.def("set_ui_message", &set_ui_message, "Set ui message");
    m.def("set_state_action_message", &set_state_action_message, "Set state action message");
    py::class_<ENVIRONMENT::Parameters>(m, "Parameters")
        .def(py::init<>())
        .def("__copy__", [](const ENVIRONMENT::Parameters &self) {
            return ENVIRONMENT::Parameters(self);
        });
    py::class_<ENVIRONMENT::State>(m, "State")
    .def(py::init<>())
    .def("__copy__", [](const ENVIRONMENT::State &self) {
        return ENVIRONMENT::State(self);
    })
    .def("assign", [](ENVIRONMENT::State &self, const ENVIRONMENT::State &other) {
        self = other;
    })
    .def_property("position", 
        [](ENVIRONMENT::State &self){
            return py::array_t<T>({3}, {sizeof(T)}, self.position, py::cast(&self));
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.position));
        }
    )
    .def_property("orientation",
        [](ENVIRONMENT::State &self){
            return py::array_t<T>({4}, {sizeof(T)}, self.orientation, py::cast(&self));
        },
        [](ENVIRONMENT::State &self, const std::array<T, 4> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.orientation));
        }
    )
    .def_property("linear_velocity",
        [](ENVIRONMENT::State &self){
            return py::array_t<T>({3}, {sizeof(T)}, self.linear_velocity, py::cast(&self));
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.linear_velocity));
        }
    )
    .def_property("angular_velocity",
        [](ENVIRONMENT::State &self){
            return py::array_t<T>({3}, {sizeof(T)}, self.angular_velocity, py::cast(&self));
        },
        [](ENVIRONMENT::State &self, const std::array<T, 3> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.angular_velocity));
        }
    )
    .def_property("rpm", 
        [](ENVIRONMENT::State &self){
            return py::array_t<T>({4}, {sizeof(T)}, self.rpm, py::cast(&self));
        },
        [](ENVIRONMENT::State &self, const std::array<T, 4> &new_data) {
            std::copy(new_data.begin(), new_data.end(), std::begin(self.rpm));
        }
    );

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_property_readonly("DIM", [](const Observation &self) { return Observation::DIM; })
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
    m.def("parameters_from_json", &parameters_from_json, "Set parameters from json");

#ifdef L2F_VECTOR
#ifndef DEBUG
    // vector_factory<1>(m);
    // vector_factory<2>(m);
    // vector_factory<4>(m);
#endif
    vector_factory<8>(m);
#ifndef DEBUG
    // vector_factory<16>(m);
    // vector_factory<32>(m);
    // vector_factory<64>(m);
    // vector_factory<128>(m);
    // vector_factory<256>(m);
    // vector_factory<512>(m);
    // vector_factory<1024>(m);
    // vector_factory<2048>(m);
    // vector_factory<4096>(m);
    // vector_factory<8192>(m);
#ifdef L2F_VECTOR_N_ENVIRONMENTS
    vector_factory<L2F_VECTOR_N_ENVIRONMENTS>(m);
#endif
#endif
#endif
}