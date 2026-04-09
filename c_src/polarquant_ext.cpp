#include <pybind11/pybind11.h>
#include <mlx/mlx.h>
#include "forge/common.h"

namespace py = pybind11;
using namespace mlx::core;

class CartesianToPolar : public mlx::core::Primitive {
    uint32_t bits;
public:
    CartesianToPolar(mlx::core::StreamOrDevice s, uint32_t bits) 
        : mlx::core::Primitive(s), bits(bits) {}

    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        throw std::runtime_error("CPU eval not implemented. Use GPU.");
    }

    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& x = inputs[0];
        auto& out = outputs[0];
        out.set_data(allocator::malloc(out.nbytes()));

        uint32_t HD = x.shape(-1);
        uint32_t B = x.size() / HD;

        mlx::metal::MTLFCList func_consts = {
            {&bits, MTL::DataTypeUInt, 0},
            {&HD, MTL::DataTypeUInt, 1}
        };

        auto kernel = forge::ForgeDispatcher::get_kernel("cartesian_to_polar", "polarquant", func_consts);
        auto& d = mlx::metal::device(mx::default_device().index);
        auto& compute_encoder = d.get_command_encoder(stream().index);

        compute_encoder.set_compute_pipeline_state(kernel);
        compute_encoder.set_input_array(x, 0);
        compute_encoder.set_output_array(out, 1);

        MTL::Size group_dims = MTL::Size(256, 1, 1);
        MTL::Size grid_dims = MTL::Size((B + 255) / 256, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    void print(std::ostream& os) override {
        os << "CartesianToPolar(bits=" << bits << ")";
    }
};

array cartesian_to_polar_fused(array x, uint32_t bits = 3, mlx::core::StreamOrDevice s = {}) {
    return array({x.shape()[0], x.shape()[1] / 2}, x.dtype(), std::make_shared<CartesianToPolar>(s, bits), {x})[0];
}

PYBIND11_MODULE(polarquant_ext, m) {
    m.doc() = "MLX C++ Extension for PolarQuant";
    m.def("cartesian_to_polar_fused", &cartesian_to_polar_fused, py::arg("x"), py::arg("stream") = py::none(), "Fused Metal implementation of Cartesian to Polar transformation");
}
