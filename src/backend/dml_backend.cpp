#include "backend/dml_backend.hpp"

#ifdef WITH_DML
#include <DirectML.h>
#include <d3d12.h>

static void
ExecuteOperator(IDMLDevice *dmlDevice, IDMLCommandRecorder *recorder,
                ID3D12GraphicsCommandList *commandList,
                IDMLCompiledOperator *compiledOperator,
                const std::vector<DML_BINDING_DESC> &inputBindings,
                const std::vector<DML_BINDING_DESC> &outputBindings) {
  ComPtr<IDMLBindingTable> bindingTable;
  DML_BINDING_TABLE_DESC tableDesc = {};
  tableDesc.Dispatchable = compiledOperator;
  tableDesc.CPUDescriptorHandle = {};
  tableDesc.GPUDescriptorHandle = {};
  tableDesc.SizeInDescriptors = 0;

  dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&bindingTable));
  bindingTable->BindInputs(static_cast<UINT>(inputBindings.size()),
                           inputBindings.data());
  bindingTable->BindOutputs(static_cast<UINT>(outputBindings.size()),
                            outputBindings.data());

  recorder->RecordDispatch(commandList, compiledOperator, bindingTable.Get());
}
#endif

DmlBuffer::DmlBuffer(size_t size_bytes, void *dml_device) : size_(size_bytes) {
#ifdef WITH_DML
  if (dml_device) {
    ID3D12Device *device = static_cast<ID3D12Device *>(dml_device);
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Width = size_bytes;
    resDesc.Height = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels = 1;
    resDesc.Format = DXGI_FORMAT_UNKNOWN;
    resDesc.SampleDesc.Count = 1;
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                    IID_PPV_ARGS(&resource_));
  }
#else
  throw std::runtime_error("DirectML is not enabled in this build.");
#endif
}

DmlBuffer::~DmlBuffer() {}

void *DmlBuffer::data() noexcept {
#ifdef WITH_DML
  return resource_.Get();
#else
  return nullptr;
#endif
}

const void *DmlBuffer::data() const noexcept {
#ifdef WITH_DML
  return resource_.Get();
#else
  return nullptr;
#endif
}

DmlBackend::DmlBackend() {
#ifdef WITH_DML
  D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                    IID_PPV_ARGS(&d3d12Device_));
  DMLCreateDevice1(d3d12Device_.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                   DML_FEATURE_LEVEL_2_0, IID_PPV_ARGS(&dmlDevice_));

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  d3d12Device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue_));

  d3d12Device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&commandAllocator_));
  d3d12Device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  commandAllocator_.Get(), nullptr,
                                  IID_PPV_ARGS(&commandList_));

  dmlDevice_->CreateCommandRecorder(IID_PPV_ARGS(&commandRecorder_));
#else
  throw std::runtime_error("DirectML is not enabled in this build.");
#endif
}

DmlBackend::~DmlBackend() {}

std::unique_ptr<DeviceBuffer> DmlBackend::allocate(size_t bytes) {
#ifdef WITH_DML
  return std::make_unique<DmlBuffer>(bytes, d3d12Device_.Get());
#else
  return std::make_unique<DmlBuffer>(bytes, nullptr);
#endif
}

void DmlBackend::copy_host_to_device(const float *host_ptr,
                                     DeviceBuffer &device_buf,
                                     size_t size_bytes) {
#ifdef WITH_DML
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resDesc.Width = size_bytes;
  resDesc.Height = 1;
  resDesc.DepthOrArraySize = 1;
  resDesc.MipLevels = 1;
  resDesc.Format = DXGI_FORMAT_UNKNOWN;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  ComPtr<ID3D12Resource> uploadBuffer;
  d3d12Device_->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer));

  void *mappedPtr = nullptr;
  uploadBuffer->Map(0, nullptr, &mappedPtr);
  std::memcpy(mappedPtr, host_ptr, size_bytes);
  uploadBuffer->Unmap(0, nullptr);

  DmlBuffer &dmlBuf = static_cast<DmlBuffer &>(device_buf);
  commandList_->CopyBufferRegion(dmlBuf.resource().Get(), 0, uploadBuffer.Get(),
                                 0, size_bytes);

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::copy_to_device(const float *host_ptr, DeviceBuffer &device_buf,
                                size_t dest_offset_bytes, size_t size_bytes) {
#ifdef WITH_DML
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resDesc.Width = size_bytes;
  resDesc.Height = 1;
  resDesc.DepthOrArraySize = 1;
  resDesc.MipLevels = 1;
  resDesc.Format = DXGI_FORMAT_UNKNOWN;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  ComPtr<ID3D12Resource> uploadBuffer;
  d3d12Device_->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer));

  void *mappedPtr = nullptr;
  uploadBuffer->Map(0, nullptr, &mappedPtr);
  std::memcpy(mappedPtr, host_ptr, size_bytes);
  uploadBuffer->Unmap(0, nullptr);

  DmlBuffer &dmlBuf = static_cast<DmlBuffer &>(device_buf);
  commandList_->CopyBufferRegion(dmlBuf.resource().Get(), dest_offset_bytes,
                                 uploadBuffer.Get(), 0, size_bytes);

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::copy_device_to_host(const DeviceBuffer &device_buf,
                                     float *host_ptr, size_t size_bytes) {
#ifdef WITH_DML
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_READBACK;

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resDesc.Width = size_bytes;
  resDesc.Height = 1;
  resDesc.DepthOrArraySize = 1;
  resDesc.MipLevels = 1;
  resDesc.Format = DXGI_FORMAT_UNKNOWN;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  ComPtr<ID3D12Resource> readbackBuffer;
  d3d12Device_->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbackBuffer));

  const DmlBuffer &dmlBuf = static_cast<const DmlBuffer &>(device_buf);
  commandList_->CopyBufferRegion(readbackBuffer.Get(), 0,
                                 dmlBuf.resource().Get(), 0, size_bytes);

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  void *mappedPtr = nullptr;
  readbackBuffer->Map(0, nullptr, &mappedPtr);
  std::memcpy(host_ptr, mappedPtr, size_bytes);
  readbackBuffer->Unmap(0, nullptr);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::fill(DeviceBuffer &buffer, float value) {
#ifdef WITH_DML
  size_t count = buffer.size() / sizeof(float);
  std::vector<float> host_data(count, value);
  copy_host_to_device(host_data.data(), buffer, buffer.size());
#endif
}

void DmlBackend::add(const DeviceBuffer &a, size_t a_offset,
                     const Shape &a_shape, const Strides &a_strides,
                     const DeviceBuffer &b, size_t b_offset,
                     const Shape &b_shape, const Strides &b_strides,
                     DeviceBuffer &out, size_t out_offset,
                     const Shape &out_shape, const Strides &out_strides) {
#ifdef WITH_DML
  DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;

  size_t count = out.size() / sizeof(float);
  UINT64 dims[] = {1, 1, 1, count};

  DML_BUFFER_TENSOR_DESC aTensorDesc = {dataType, flags, 4,         dims,
                                        nullptr,  0,     out.size()};
  DML_TENSOR_DESC aDesc = {DML_TENSOR_TYPE_BUFFER, &aTensorDesc};

  DML_BUFFER_TENSOR_DESC bTensorDesc = {dataType, flags, 4,         dims,
                                        nullptr,  0,     out.size()};
  DML_TENSOR_DESC bDesc = {DML_TENSOR_TYPE_BUFFER, &bTensorDesc};

  DML_BUFFER_TENSOR_DESC outTensorDesc = {dataType, flags, 4,         dims,
                                          nullptr,  0,     out.size()};
  DML_TENSOR_DESC outDesc = {DML_TENSOR_TYPE_BUFFER, &outTensorDesc};

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc = {&aDesc, &bDesc, &outDesc};
  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc};

  ComPtr<IDMLOperator> op;
  dmlDevice_->CreateOperator(&opDesc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiledOp;
  dmlDevice_->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiledOp));

  const DmlBuffer &dmlA = static_cast<const DmlBuffer &>(a);
  const DmlBuffer &dmlB = static_cast<const DmlBuffer &>(b);
  DmlBuffer &dmlOut = static_cast<DmlBuffer &>(out);

  DML_BUFFER_BINDING aBinding = {dmlA.resource().Get(),
                                 a_offset * sizeof(float), out.size()};
  DML_BINDING_DESC aBindDesc = {DML_BINDING_TYPE_BUFFER, &aBinding};

  DML_BUFFER_BINDING bBinding = {dmlB.resource().Get(),
                                 b_offset * sizeof(float), out.size()};
  DML_BINDING_DESC bBindDesc = {DML_BINDING_TYPE_BUFFER, &bBinding};

  DML_BUFFER_BINDING outBinding = {dmlOut.resource().Get(),
                                   out_offset * sizeof(float), out.size()};
  DML_BINDING_DESC outBindDesc = {DML_BINDING_TYPE_BUFFER, &outBinding};

  ExecuteOperator(dmlDevice_.Get(), commandRecorder_.Get(), commandList_.Get(),
                  compiledOp.Get(), {aBindDesc, bBindDesc}, {outBindDesc});

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::sub(const DeviceBuffer &, size_t, const Shape &,
                     const Strides &, const DeviceBuffer &, size_t,
                     const Shape &, const Strides &, DeviceBuffer &, size_t,
                     const Shape &, const Strides &) {}

void DmlBackend::mul(const DeviceBuffer &, size_t, const Shape &,
                     const Strides &, const DeviceBuffer &, size_t,
                     const Shape &, const Strides &, DeviceBuffer &, size_t,
                     const Shape &, const Strides &) {}

void DmlBackend::div(const DeviceBuffer &, size_t, const Shape &,
                     const Strides &, const DeviceBuffer &, size_t,
                     const Shape &, const Strides &, DeviceBuffer &, size_t,
                     const Shape &, const Strides &) {}

void DmlBackend::matmul(const DeviceBuffer &a, size_t a_offset,
                        const Shape &a_shape, const Strides &a_strides,
                        const DeviceBuffer &b, size_t b_offset,
                        const Shape &b_shape, const Strides &b_strides,
                        DeviceBuffer &out, size_t out_offset,
                        const Shape &out_shape, const Strides &out_strides) {
#ifdef WITH_DML
  DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;

  size_t M = a_shape[0];
  size_t K = a_shape[1];
  size_t N = b_shape[1];

  UINT64 aDims[] = {1, 1, M, K};
  DML_BUFFER_TENSOR_DESC aTensorDesc = {dataType, flags, 4,       aDims,
                                        nullptr,  0,     a.size()};
  DML_TENSOR_DESC aDesc = {DML_TENSOR_TYPE_BUFFER, &aTensorDesc};

  UINT64 bDims[] = {1, 1, K, N};
  DML_BUFFER_TENSOR_DESC bTensorDesc = {dataType, flags, 4,       bDims,
                                        nullptr,  0,     b.size()};
  DML_TENSOR_DESC bDesc = {DML_TENSOR_TYPE_BUFFER, &bTensorDesc};

  UINT64 outDims[] = {1, 1, M, N};
  DML_BUFFER_TENSOR_DESC outTensorDesc = {dataType, flags, 4,         outDims,
                                          nullptr,  0,     out.size()};
  DML_TENSOR_DESC outDesc = {DML_TENSOR_TYPE_BUFFER, &outTensorDesc};

  DML_GEMM_OPERATOR_DESC gemmDesc = {
      &aDesc, &bDesc, nullptr, &outDesc, DML_OPERATOR_FLAG_NONE, 1.0f, 0.0f};
  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_GEMM, &gemmDesc};

  ComPtr<IDMLOperator> op;
  dmlDevice_->CreateOperator(&opDesc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiledOp;
  dmlDevice_->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiledOp));

  const DmlBuffer &dmlA = static_cast<const DmlBuffer &>(a);
  const DmlBuffer &dmlB = static_cast<const DmlBuffer &>(b);
  DmlBuffer &dmlOut = static_cast<DmlBuffer &>(out);

  DML_BUFFER_BINDING aBinding = {dmlA.resource().Get(),
                                 a_offset * sizeof(float), a.size()};
  DML_BINDING_DESC aBindDesc = {DML_BINDING_TYPE_BUFFER, &aBinding};

  DML_BUFFER_BINDING bBinding = {dmlB.resource().Get(),
                                 b_offset * sizeof(float), b.size()};
  DML_BINDING_DESC bBindDesc = {DML_BINDING_TYPE_BUFFER, &bBinding};

  DML_BUFFER_BINDING outBinding = {dmlOut.resource().Get(),
                                   out_offset * sizeof(float), out.size()};
  DML_BINDING_DESC outBindDesc = {DML_BINDING_TYPE_BUFFER, &outBinding};

  ExecuteOperator(dmlDevice_.Get(), commandRecorder_.Get(), commandList_.Get(),
                  compiledOp.Get(), {aBindDesc, bBindDesc}, {outBindDesc});

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::sum(const DeviceBuffer &, size_t, const Shape &,
                     const Strides &, DeviceBuffer &, size_t, const Shape &,
                     const Strides &, const std::vector<size_t> &) {}

void DmlBackend::relu(const DeviceBuffer &in, size_t in_offset,
                      const Shape &in_shape, const Strides &in_strides,
                      DeviceBuffer &out, size_t out_offset,
                      const Shape &out_shape, const Strides &out_strides) {
#ifdef WITH_DML
  DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;

  size_t count = out.size() / sizeof(float);
  UINT64 dims[] = {1, 1, 1, count};

  DML_BUFFER_TENSOR_DESC inTensorDesc = {dataType, flags, 4,         dims,
                                         nullptr,  0,     out.size()};
  DML_TENSOR_DESC inDesc = {DML_TENSOR_TYPE_BUFFER, &inTensorDesc};

  DML_BUFFER_TENSOR_DESC outTensorDesc = {dataType, flags, 4,         dims,
                                          nullptr,  0,     out.size()};
  DML_TENSOR_DESC outDesc = {DML_TENSOR_TYPE_BUFFER, &outTensorDesc};

  DML_ACTIVATION_RELU_OPERATOR_DESC reluDesc = {&inDesc, &outDesc};
  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_ACTIVATION_RELU, &reluDesc};

  ComPtr<IDMLOperator> op;
  dmlDevice_->CreateOperator(&opDesc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiledOp;
  dmlDevice_->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiledOp));

  const DmlBuffer &dmlIn = static_cast<const DmlBuffer &>(in);
  DmlBuffer &dmlOut = static_cast<DmlBuffer &>(out);

  DML_BUFFER_BINDING inBinding = {dmlIn.resource().Get(),
                                  in_offset * sizeof(float), out.size()};
  DML_BINDING_DESC inBindDesc = {DML_BINDING_TYPE_BUFFER, &inBinding};

  DML_BUFFER_BINDING outBinding = {dmlOut.resource().Get(),
                                   out_offset * sizeof(float), out.size()};
  DML_BINDING_DESC outBindDesc = {DML_BINDING_TYPE_BUFFER, &outBinding};

  ExecuteOperator(dmlDevice_.Get(), commandRecorder_.Get(), commandList_.Get(),
                  compiledOp.Get(), {inBindDesc}, {outBindDesc});

  commandList_->Close();
  ID3D12CommandList *lists[] = {commandList_.Get()};
  commandQueue_->ExecuteCommandLists(1, lists);

  ComPtr<ID3D12Fence> fence;
  d3d12Device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
  commandQueue_->Signal(fence.Get(), 1);
  HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  WaitForSingleObject(event, INFINITE);
  CloseHandle(event);

  commandAllocator_->Reset();
  commandList_->Reset(commandAllocator_.Get(), nullptr);
#endif
}

void DmlBackend::relu_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {}

void DmlBackend::sigmoid(const DeviceBuffer &, size_t, const Shape &,
                         const Strides &, DeviceBuffer &, size_t, const Shape &,
                         const Strides &) {}

void DmlBackend::sigmoid_backward(const DeviceBuffer &, size_t, const Shape &,
                                  const Strides &, const DeviceBuffer &, size_t,
                                  const Shape &, const Strides &,
                                  DeviceBuffer &, size_t, const Shape &,
                                  const Strides &) {}

void DmlBackend::tanh(const DeviceBuffer &, size_t, const Shape &,
                      const Strides &, DeviceBuffer &, size_t, const Shape &,
                      const Strides &) {}

void DmlBackend::tanh_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {}

void DmlBackend::softmax(const DeviceBuffer &, size_t, const Shape &,
                         const Strides &, DeviceBuffer &, size_t, const Shape &,
                         const Strides &, size_t) {}

void DmlBackend::softmax_backward(const DeviceBuffer &, size_t, const Shape &,
                                  const Strides &, const DeviceBuffer &, size_t,
                                  const Shape &, const Strides &,
                                  DeviceBuffer &, size_t, const Shape &,
                                  const Strides &, size_t) {}

void DmlBackend::sqrt(const DeviceBuffer &, size_t, const Shape &,
                      const Strides &, DeviceBuffer &, size_t, const Shape &,
                      const Strides &) {}

void DmlBackend::sqrt_backward(const DeviceBuffer &, size_t, const Shape &,
                               const Strides &, const DeviceBuffer &, size_t,
                               const Shape &, const Strides &, DeviceBuffer &,
                               size_t, const Shape &, const Strides &) {}

void DmlBackend::log(const DeviceBuffer &, size_t, const Shape &,
                     const Strides &, DeviceBuffer &, size_t, const Shape &,
                     const Strides &) {}

void DmlBackend::log_backward(const DeviceBuffer &, size_t, const Shape &,
                              const Strides &, const DeviceBuffer &, size_t,
                              const Shape &, const Strides &, DeviceBuffer &,
                              size_t, const Shape &, const Strides &) {}

void DmlBackend::conv2d(const DeviceBuffer &, size_t, const Shape &,
                        const Strides &, const DeviceBuffer &, size_t,
                        const Shape &, const Strides &, const DeviceBuffer &,
                        size_t, const Shape &, const Strides &, DeviceBuffer &,
                        size_t, const Shape &, const Strides &, size_t,
                        size_t) {}

void DmlBackend::conv2d_backward(const DeviceBuffer &, size_t, const Shape &,
                                 const Strides &, const DeviceBuffer &, size_t,
                                 const Shape &, const Strides &,
                                 const DeviceBuffer &, size_t, const Shape &,
                                 const Strides &, DeviceBuffer &, size_t,
                                 const Shape &, const Strides &, DeviceBuffer &,
                                 size_t, const Shape &, const Strides &,
                                 DeviceBuffer &, size_t, const Shape &,
                                 const Strides &, size_t, size_t) {}

void DmlBackend::maxpool2d(const DeviceBuffer &, size_t, const Shape &,
                           const Strides &, DeviceBuffer &, size_t,
                           const Shape &, const Strides &, size_t, size_t,
                           size_t) {}

void DmlBackend::maxpool2d_backward(const DeviceBuffer &, size_t, const Shape &,
                                    const Strides &, const DeviceBuffer &,
                                    size_t, const Shape &, const Strides &,
                                    const DeviceBuffer &, size_t, const Shape &,
                                    const Strides &, DeviceBuffer &, size_t,
                                    const Shape &, const Strides &, size_t,
                                    size_t, size_t) {}

void DmlBackend::embedding(const DeviceBuffer &, size_t, const Shape &,
                           const Strides &, const DeviceBuffer &, size_t,
                           const Shape &, const Strides &, DeviceBuffer &,
                           size_t, const Shape &, const Strides &) {}

void DmlBackend::embedding_backward(const DeviceBuffer &, size_t, const Shape &,
                                    const Strides &, const DeviceBuffer &,
                                    size_t, const Shape &, const Strides &,
                                    DeviceBuffer &, size_t, const Shape &,
                                    const Strides &) {}

void DmlBackend::slice_backward(const DeviceBuffer &, size_t, const Shape &,
                                const Strides &, DeviceBuffer &, size_t,
                                const Shape &, const Strides &, size_t,
                                size_t) {}
