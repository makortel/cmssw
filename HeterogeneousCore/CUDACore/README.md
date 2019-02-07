# Prototype for CMSSW interface to CUDA algorithms

## Outline

* [Introduction](introduction)
  * [Design goals]()
  * [Overall guidelines]()
* [Sub-packages]()
* Examples
  * Isolated producer (no CUDA input nor output)
  * Producer with CUDA input
  * Producer with CUDA output
  * Produder with CUDA input and output (with ExternalWork)
  * Producer with CUDA input and output (without ExternalWork)
  * Configuration
* More details


## Introduction

This page documents the CUDA integration within CMSSW

### Design goals

1. Provide a mechanism for a chain of modules to share a resource
   * Resource can be e.g. CUDA device memory or a CUDA stream
2. Minimize data movements between the CPU and the device
3. Support multi devices
4. Allow the same job configuration to be used on all hardware combinations

### Overall guidelines

1. Within the `acquire()`/`produce()` functions all CUDA operations should be asynchronous, i.e.
   * Use `cudaMemcpyAsync()`, `cudaMemsetAsync()`, `cudaMemPrefetchAsync()` etc.
   * Avoid `cudaMalloc*()`, `cudaHostAlloc()`, `cudaFree*()`, `cudaHostRegister()`, `cudaHostUnregister()` on every event
     * Occasional calls are permitted through a caching allocator
   * Avoid `assert()` in device functions, or use `#include HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h`
     * With the latter the `assert()`s in CUDA code are disabled by
       default, but can be enabled by defining `GPU_DEBUG` macro
       (before the aforementioned include)
2. Synchronization needs should be fulfilled with
   [`ExternalWork`](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#edm_ExternalWork)
   extension to EDProducers
   * `ExternalWork` can be used to replace one synchronization point
     (e.g. between device kernels and copying a known amount of data
     back to CPU).
   * For further synchronization points (e.g. copying data whose
     amount is known only at the device side), split the work to
     multiple `ExternalWork` producers. This approach has the added
     benefit that e.g. data transfers to CPU become on-demand automatically
   * A general breakdown of the possible steps:
     * Convert input legacy CPU data format to CPU SoA
     * Transfer input CPU SoA to GPU
     * Run kernels
     * Transfer the number of output elements to CPU
     * Transfer the output data from GPU to CPU SoA
     * Convert the output SoA to legacy GPU data formats
3. Within `acquire()`/`produce()`, the CUDA device is set implicitly
   and the CUDA stream is provided by the system (with
   `CUDAScopedContext`)
   * It is strongly recommended to use the provided CUDA stream for all operations
     * If that is not feasible for some reason, the provided CUDA
       stream must synchronize with the work queued on other CUDA
       streams (with CUDA events and `cudaStreamWaitEvent()`)
4. Outside of `acquire()`/`produce()`, CUDA API functions may be
   called only if `CUDAService::enabled()` returns `true`.
   * With point 3 it follows that in these cases multiple devices have
     to be dealt with explicitly, as well as CUDA streams

## Sub-packages
* [`HeterogeneousCore/CUDACore`](#cuda-integration) CUDA-specific core components
* [`HeterogeneousCore/CUDAServices`](../CUDAServices) Various edm::Services related to CUDA
* [`HeterogeneousCore/CUDAUtilities`](../CUDAUtilities) Various utilities for CUDA kernel code
* [`HeterogeneousCore/CUDATest`](../CUDATest) Test modules and configurations
* [`CUDADataFormats/Common`](../../CUDADataFormats/Common) Utilities for event products with CUDA data

## Examples

### Isolated producer (no CUDA input nor output)

```cpp
class IsolatedProducerCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  ...
  IsolatedProducerGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<InputData> inputToken_;
  edm::EDPutTokenT<OutputData> outputToken_;
};
...
void IsolatedProducerCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Sets the current device and creates a CUDA stream
  CUDAScopedContext ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  auto const& inputData = iEvent.get(inputToken_);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by CUDAScopedContext::stream()
  gpuAlgo_.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void IsolatedProducerCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Real life is likely more complex than this simple example. Here
  // getResult() returns some data in CPU memory that is passed
  // directly to the OutputData constructor.
  iEvent.emplace(outputToken_, gpuAlgo_.getResult());
}
```

### Producer with CUDA output

```cpp
class ProducerOutputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  ...
  ProducerOutputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<InputData> inputToken_;
  edm::EDPutTokenT<CUDAProduct<OutputData>> outputToken_;
  CUDAContextToken ctxTmp_;
};
...
void ProducerOutputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  // Sets the current device and creates a CUDA stream
  CUDAScopedContext ctx{iEvent.streamID(), std::move(waitingTaskHolder)};

  auto const& inputData = iEvent.get(inputToken_);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by CUDAScopedContext::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Passes the current device and CUDA stream to produce()
  // Feels a bit silly, and will hopefully get improved in the future
  ctxTmp_ = ctx.toToken();

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void ProducerOutputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Sets again the current device, uses the CUDA stream created in the acquire()
  CUDAScopedContext ctx{std::move(ctxTmp_)};

  // Now getResult() returns data in GPU memory that is passed to the
  // constructor of OutputData. CUDAScopedContext::emplace() wraps the
  // OutputData to CUDAProduct<OutputData>. CUDAProduct<T> stores also
  // the current device and the CUDA stream since those will be needed
  // in the consumer side.
  ctx.emplace(iEvent, outputToken_, gpuAlgo.getResult());
}
```

### Producer with CUDA input

```cpp
class ProducerInputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<CUDAProduct<InputData>> inputToken_;
  edm::EDPutTokenT<OutputData> outputToken_;
};
...
void ProducerInputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  CUDAProduct<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and also use the same CUDA stream
  CUDAScopedContext ctx{inputDataWrapped, std::move(waitingTaskHolder)};

  // Alternatively, if e.g. there is another module queuing //
  // independent work to the CUDA stream, a new CUDA stream can also be
  // created here with
  CUDAScopedContext ctx{iEvent.streamID(), std::move(waitingTaskHolder);
  
  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the CUDAScopedContext holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by CUDAScopedContext::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void ProducerInputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Real life is likely more complex than this simple example. Here
  // getResult() returns some data in CPU memory that is passed
  // directly to the OutputData constructor.
  iEvent.emplace(outputToken_, gpuAlgo_.getResult());
}
```

### Producer with CUDA input and output (with ExternalWork)

```cpp
class ProducerInputOutputCUDA: public edm::stream::EDProducer<ExternalWork> {
public:
  ...
  void acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup& iSetup) override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<CUDAProduct<InputData>> inputToken_;
  edm::EDPutTokenT<CUDAProduct<OutputData>> outputToken_;
  CUDAContextToken ctxTmp_;
};
...
void ProducerInputOutputCUDA::acquire(edm::Event const& iEvent, edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  CUDAProduct<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and also use the same CUDA stream
  CUDAScopedContext ctx{inputDataWrapped, std::move(waitingTaskHolder)};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the CUDAScopedContext holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by CUDAScopedContext::stream()
  gpuAlgo.makeAsync(inputData, ctx.stream());

  // Passes the current device and CUDA stream to produce()
  // Feels a bit silly, and will hopefully get improved in the future
  ctxTmp_ = ctx.toToken();

// Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}

// Called after the asynchronous work has finished
void ProducerInputOutputCUDA::produce(edm::Event& iEvent, edm::EventSetup& iSetup) {
  // Sets again the current device, uses the CUDA stream created in the acquire()
  CUDAScopedContext ctx{std::move(ctxTmp_)};

  // Now getResult() returns data in GPU memory that is passed to the
  // constructor of OutputData. CUDAScopedContext::emplace() wraps the
  // OutputData to CUDAProduct<OutputData>. CUDAProduct<T> stores also
  // the current device and the CUDA stream since those will be needed
  // in the consumer side.
  ctx.emplace(iEvent, outputToken_, gpuAlgo.getResult());
}
```

### Producer with CUDA input and output (without ExternalWork)

If the producer does not need to transfer anything back to CPU (like
the number of output elements), the `ExternalWork` extension is not
needed as there is no need to synchronize.

```cpp
class ProducerInputOutputCUDA: public edm::global::EDProducer<> {
public:
  ...
  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup& iSetup) const override;
  ...
private:
  ...
  ProducerInputGPUAlgo gpuAlgo_;
  edm::EDGetTokenT<CUDAProduct<InputData>> inputToken_;
  edm::EDPutTokenT<CUDAProduct<OutputData>> outputToken_;
};
...
void ProducerInputOutputCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup& iSetup) const {
  CUDAProduct<InputData> const& inputDataWrapped = iEvent.get(inputToken_);

  // Set the current device to the same that was used to produce
  // InputData, and also use the same CUDA stream
  CUDAScopedContext ctx{streamID};

  // Grab the real input data. Checks that the input data is on the
  // current device. If the input data was produced in a different CUDA
  // stream than the CUDAScopedContext holds, create an inter-stream
  // synchronization point with CUDA event and cudaStreamWaitEvent()
  auto const& inputData = ctx.get(inputDataWrapped);

  // Queues asynchronous data transfers and kernels to the CUDA stream
  // returned by CUDAScopedContext::stream(). Here makeAsync() also
  // returns data in GPU memory that is passed to the constructor of
  // OutputData. CUDAScopedContext::emplace() wraps the OutputData to
  // CUDAProduct<OutputData>. CUDAProduct<T> stores also the current
  // device and the CUDA stream since those will be needed in the
  // consumer side.
  ctx.emplace(iEvent, outputToken, gpuAlgo.makeAsync(inputData, ctx.stream());

  // Destructor of ctx queues a callback to the CUDA stream notifying
  // waitingTaskHolder when the queued asynchronous work has finished
}
```

### Configuration

```python
```


##################################################

## Choosing device

### GPU and CPU

Currently the device type choice (CPU vs. GPU) is done at the
configuration level with `cms.Modifier`. In the near future this will
be changed to a decision made at the beginning of the job with a
[`SwitchProducer`](https://github.com/cms-sw/cmssw/pull/25439).

For multi-GPU setup the device is chosen in the first CUDA module in a
chain of modules by one of the constructors of
`CUDAScopedContext`
```cpp
auto ctx = CUDAScopedContext(iEvent.streamID());
```
As the choice is still the static EDM stream to device assignment, the
EDM stream ID is needed. The logic will likely evolve in the future.

### Always on GPU

In case the chain of modules should always be run on a GPU, the
configuration should be built only with the GPU modules.


## Data model

The GPU data should be a class/struct containing smart pointer(s) to
device data (see [Memory allocation](#memory-allocation)). When
putting the data to event, the data is wrapped to `CUDA<T>` template,
which holds
* the GPU data
  * must be movable, but no other restrictions
* the current device where the data was produced, and the CUDA stream the data was produced with
* [CUDA event for synchronization between multiple CUDA streams](#synchronizing-between-cuda-streams)

Note that the `CUDA<T>` wrapper can be constructed only with
`CUDAScopedContext::wrap()`, and the data `T` can be obtained from it
only with `CUDAScopedContext::get()`, as described further below. When
putting the data product directly to `edm::Event`, also
`CUDASCopedContext::emplace()` can be used.

## CUDA EDProducer

### Class declaration

The CUDA producers are normal EDProducers. Contrary to
`HeterogeneousEDProducer`, the `ExternalWork` extension is **not**
required. Its use is recommended though when transferring data from
GPU to CPU.

### Memory allocation

The memory allocations should be done dynamically with `CUDAService`
```cpp
edm::Service<CUDAService> cs;
edm::cuda::device::unique_ptr<float[]> device_buffer = cs->make_device_unique<float[]>(50, cudaStream);
edm::cuda::host::unique_ptr<float[]>   host_buffer   = cs->make_host_unique<float[]>(50, cudaStream);
```

in the `acquire()` and `produce()` functions. The same
`cuda::stream_t<>` object that is used for transfers and kernels
should be passed to the allocator.

The allocator is based on `cub::CachingDeviceAllocator`. The memory is
guaranteed to be reserved
* for the host: up to the destructor of the `unique_ptr`
* for the device: until all work queued in the `cudaStream` up to the point when the `unique_ptr` destructor is called has finished

### Setting the current device

A CUDA producer should construct `CUDAScopedContext` in `acquire()`
either with `edm::StreamID`, or with a `CUDA<T>` read as an input.

A CUDA producer should read either `CUDAToken` (from
`CUDADeviceChooser`) or one or more `CUDA<T>` products. Then, in the
`acquire()`/`produce()`, it should construct `CUDAScopedContext` from
one of them
```cpp
// From edm::StreamID
auto ctx = CUDAScopedContext(iEvent.streamID());

/// From CUDA<T>
edm::Handle<CUDA<GPUClusters>> handle;
iEvent.getByToken(srctoken_, handle);
auto ctx = CUDAScopedContext(*handle);
```

`CUDAScopedContext` works in the RAII way and does the following
* Sets the current device for the current scope
  - If constructed from the `edm::StreamID`, makes the device choice and creates a new CUDA stream
  - If constructed from the `CUDA<T>`, uses the same device and CUDA stream as was used to produce the `CUDA<T>`
* Gives access to the CUDA stream the algorithm should use to queue asynchronous work
* Calls `edm::WaitingTaskWithArenaHolder::doneWaiting()` when necessary
* [Synchronizes between CUDA streams if necessary](#synchronizing-between-cuda-streams)
* Needed to get/put `CUDA<T>` from/to the event

In case of multiple input products, from possibly different CUDA
streams and/or CUDA devices, this approach gives the developer full
control in which of them the kernels of the algorithm should be run.

### Getting input

The real product (`T`) can be obtained from `CUDA<T>` only with the
help of `CUDAScopedContext`. 

```cpp
edm::Handle<CUDA<GPUClusters>> hclus;
iEvent.getByToken(srctoken_, hclus);
GPUClusters const& clus = ctx.get(*hclus);
```

This step is needed to
* check that the data are on the same CUDA device
  * if not, throw an exception (with unified memory could prefetch instead)
* if the CUDA streams are different, synchronize between them

### Calling the CUDA kernels

There is nothing special, except the CUDA stream should be obtained from
the `CUDAScopedContext`

```cpp
gpuAlgo.makeClustersAsync(..., ctx.stream());
```

### Putting output

The GPU data needs to be wrapped to `CUDA<T>` template with
`CUDAScopedContext::wrap()` or `CUDAScopedContext::emplace()`

```cpp
GPUClusters clusters = gpuAlgo.makeClustersAsync(..., ctx.stream());
std::unique_ptr<CUDA<GPUClusters>> ret = ctx.wrap(clusters);
iEvent.put(std::move(ret));

// or with one line
iEvent.put(ctx.wrap(gpuAlgo.makeClustersAsync(ctx.stream())));

// or avoid one unique_ptr with emplace
edm::PutTokenT<CUDA<GPUClusters>> putToken_ = produces<CUDA<GPUClusters>>(); // in constructor
ctx.emplace(iEvent, putToken_, gpuAlgo.makeClustersAsync(ctx.stream()));
```

This step is needed to
* store the current device and CUDA stream into `CUDA<T>`
* record the CUDA event needed for CUDA stream synchronization

### `ExternalWork` extension

Everything above works both with and without `ExternalWork`.

Without `ExternalWork` the `EDProducer`s act similar to TBB
flowgraph's "streaming node". In other words, they just queue more
asynchronous work in their `produce()`.

The `ExternalWork` is needed when one would otherwise call
`cudeStreamSynchronize()`. For example transferring something to CPU
needed for downstream DQM, or queueing more asynchronous work. With
`ExternalWork` an `acquire()` method needs to be implemented that gets
an `edm::WaitingTaskWithArenaHolder` parameter. The
`WaitingTaskWithArenaHolder` should then be passed to the constructor
of `CUDAScopedContext` along

```cpp
void acquire(..., edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::Handle<CUDA<GPUClusters>> handle;
  iEvent.getByToken(token_, handle);
  auto ctx = CUDAScopedContext(*handle, std::move(waitingTaskHolder)); // can also copy instead of move if waitingTaskHolder is needed for something else as well
  ...
```

When constructed this way, `CUDAScopedContext` registers a callback
function to the CUDA stream in its destructor to call
`waitingTaskHolder.doneWaiting()`.

A GPU->GPU producer needs a `CUDAScopedContext` also in its
`produce()`. Currently the best way is to store the state of
`CUDAScopedContext` to `CUDAContextToken` member variable:

```cpp
class FooProducerCUDA ... {
  ...
  CUDAContextToken ctxTmp_;
};

void acquire(...) {
  ...
  ctxTmp_ = ctx.toToken();
}

void produce(...( {
  ...
  auto ctx = CUDAScopedContext(std::move(ctxTmp_));
}
```

Ideas for improvements are welcome.


### Transferring GPU data to CPU

The GPU->CPU data transfer needs synchronization to ensure the CPU
memory to have all data before putting that to the event. This means
the `ExternalWork` needs to be used along
* In `acquire()`
  * (allocate CPU memory buffers)
  * Queue all GPU->CPU transfers asynchronously
* In `produce()`
  * If needed, read additional CPU products (e.g. from `edm::Ref`s)
  * Reformat data back to legacy data formats
  * Note: `CUDAScopedContext` is **not** needed in `produce()`

### Synchronizing between CUDA streams

In case the producer needs input data that were produced in two (or
more) CUDA streams, these streams have to be synchronized (since CMSSW
framework no longer guarantees the synchronization as was the case
with `HeterogeneousEDProducer`). Here this synchronization is achieved
with CUDA events.

Each `CUDA<T>` constains also a CUDA event object. The call to
`CUDAScopedContext::wrap()` will *record* the event in the CUDA stream.
This means that when all work queued to the CUDA stream up to that
point has been finished, the CUDA event becomes *occurred*. Then, in
`CUDAScopedContext::get()`, if the `CUDA<T>` to get from has a
different CUDA stream than the `CUDAScopedContext`,
`cudaStreamWaitEvent(stream, event)` is called. This means that all
subsequent work queued to the CUDA stream will wait for the CUDA event
to become occurred. Therefore this subsequent work can assume that the
to-be-getted CUDA product exists.

## Configuration

### With `cms.Modifier`

```python
process.foo = cms.EDProducer("FooProducer") # legacy CPU

from Configuration.ProcessModifiers.gpu_cff import gpu
process.fooCUDA = cms.EDProducer("FooProducerCUDA")
gpu.toReplaceWith(process.foo, cms.EDProducer("FooProducerFromCUDA", src="fooCUDA"))

process.fooTaskCUDA = cms.Task(process.fooCUDA)
process.fooTask = cms.Task(
    process.foo,
    process.fooTaskCUDA
)
```

For a more complete example, see [here](../CUDATests/test/testCUDA_cfg.py).

### With `SwitchProducer`

```python
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
process.foo = SwitchProducerCUDA(
    cpu = cms.EDProducer("FooProducer"), # legacy CPU
    cuda = cms.EDProducer("FooProducerFromCUDA", src="fooCUDA")
)
process.fooCUDA = cms.EDProducer("FooProducerCUDA")

process.fooTaskCUDA = cms.Task(process.fooCUDA)
process.fooTask = cms.Task(
    process.foo,
    process.fooTaskCUDA
)
```

# Extension to other devices

The C++ side extends in a straightforward way. One has to add classes
similar to `CUDAToken`, `CUDA<T>`, and `CUDAScopedContext`. Of course,
much depends on the exact details. The python configuration side
extends as well.
