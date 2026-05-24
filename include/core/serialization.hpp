#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "core/tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace serialization {

void save_parameters(const std::vector<std::shared_ptr<Tensor>> &params,
                     const std::string &filepath);

void load_parameters(const std::vector<std::shared_ptr<Tensor>> &params,
                     const std::string &filepath);

} // namespace serialization

#endif // SERIALIZATION_HPP
