#include "core/serialization.hpp"
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace serialization {

void save_parameters(const std::vector<std::shared_ptr<Tensor>> &params,
                     const std::string &filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  out.write("NETW", 4);

  uint16_t version = 1;
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));

  uint32_t count = static_cast<uint32_t>(params.size());
  out.write(reinterpret_cast<const char *>(&count), sizeof(count));

  for (const auto &tensor : params) {
    if (!tensor) {
      throw std::runtime_error("Cannot serialize a null tensor pointer.");
    }
    const auto &shape = tensor->shape();
    uint32_t dims = static_cast<uint32_t>(shape.size());
    out.write(reinterpret_cast<const char *>(&dims), sizeof(dims));

    for (size_t d : shape) {
      uint64_t dim_val = static_cast<uint64_t>(d);
      out.write(reinterpret_cast<const char *>(&dim_val), sizeof(dim_val));
    }

    size_t total_elements = count_elements(shape);
    uint64_t num_elements = static_cast<uint64_t>(total_elements);
    out.write(reinterpret_cast<const char *>(&num_elements),
              sizeof(num_elements));

    std::vector<float> host_data = tensor->to_host();
    out.write(reinterpret_cast<const char *>(host_data.data()),
              static_cast<std::streamsize>(total_elements * sizeof(float)));
  }
}

void load_parameters(const std::vector<std::shared_ptr<Tensor>> &params,
                     const std::string &filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filepath);
  }

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || std::string(magic, 4) != "NETW") {
    throw std::runtime_error("Invalid file format signature.");
  }

  uint16_t version = 0;
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (version != 1) {
    throw std::runtime_error("Unsupported serialization file version: " +
                             std::to_string(version));
  }

  uint32_t count = 0;
  in.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (count != params.size()) {
    throw std::runtime_error("Tensor count mismatch (file has " +
                             std::to_string(count) + ", model has " +
                             std::to_string(params.size()) + ").");
  }

  for (const auto &tensor : params) {
    if (!tensor) {
      throw std::runtime_error("Destination tensor pointer is null.");
    }
    uint32_t dims = 0;
    in.read(reinterpret_cast<char *>(&dims), sizeof(dims));

    const auto &dest_shape = tensor->shape();
    if (dims != dest_shape.size()) {
      throw std::runtime_error("Tensor dimension count mismatch.");
    }

    for (size_t i = 0; i < dims; ++i) {
      uint64_t dim_val = 0;
      in.read(reinterpret_cast<char *>(&dim_val), sizeof(dim_val));
      if (dim_val != static_cast<uint64_t>(dest_shape[i])) {
        throw std::runtime_error("Tensor shape mismatch at dimension " +
                                 std::to_string(i));
      }
    }

    uint64_t num_elements = 0;
    in.read(reinterpret_cast<char *>(&num_elements), sizeof(num_elements));
    size_t total_elements = count_elements(dest_shape);
    if (num_elements != static_cast<uint64_t>(total_elements)) {
      throw std::runtime_error("Tensor element count mismatch.");
    }

    std::vector<float> host_data(total_elements);
    in.read(reinterpret_cast<char *>(host_data.data()),
            static_cast<std::streamsize>(total_elements * sizeof(float)));

    tensor->copy_from_host(host_data);
  }
}

} // namespace serialization
