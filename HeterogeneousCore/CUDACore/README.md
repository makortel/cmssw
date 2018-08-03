# Next iteration of the prototype for CMSSW interface to heterogeneous algorithms

## Introduction

The current prototype with `HeterogeneousEDProducer` and
`HeterogeneousProduct` is documented [here](../Producer/README.md).
The main differences wrt. that are
* Split device-specific code to different EDProducers
* Plug components together in the configuration

This page documents the CUDA integration, and discusses briefly on how
to extend to other devices. It will be extended if/when it gets
deployed and `HeterogeneousEDProducer` retired.

## Sub-packages
* [`CUDACore`](#cuda-integration) CUDA-specific core components
* [`CUDAServices`](../CUDAServices) Various edm::Services related to CUDA
* [`CUDAUtilities`](../CUDAUtilities) Various utilities for CUDA kernel code

# CUDA integration

## Choosing device

### Dynamically between GPU and CPU

The device choosing (CPU vs. GPU, which GPU) logic is done by an
EDFilter and using Paths in the configuration.

First, a `CUDADeviceChooserFilter` EDFilter is run. It has the logic
to device whether the following chain of EDModules should run on a
CUDA device or not, and if yes, on which CUDA device. If it decides
"yes", it returns `true` and produces a `CUDAToken`, which contains
the device id and a CUDA stream. If it decides "no", it returns
`false` and does not produce anything.

Then, the pieces need to be put together in the configuration. The
`CUDADeviceChooserFilter` should be put as the first module on a
`cms.Path`, followed by the CUDA EDProducers (in the future it may
become sufficient to have only the first EDProducer of a chain in the
`Path`).
```python
process.fooCUDADeviceFilter = cms.EDFilter("CUDADeviceChooserFilter",
    src = cms.InputTag("fooCUDADevice")
)
process.fooCUDA = cms.EDProducer("FooProducerCUDA")
process.fooPathCUDA = cms.Path(
    process.fooCUDADeviceFilter + process.fooCUDA
)
```

### Always on GPU

In case the chain of modules should always be run on a GPU, the
EDFilter and Paths are not needed. In this case, a
`CUDADeviceChooserProducer` should be used to produce the `CUDAToken`.
If the machine has no GPUs or `CUDAService` is disabled, the producer
throws an exception.


## Data model

The GPU data can be a single pointer to device data, or a class/struct
containing such pointers (among other stuff). When putting the data to
event, the data is wrapped to `CUDA<T>` template, which holds
* the GPU data
  * must be movable, but no other restrictions (except need to be able to generate ROOT dictionaries from it)
* the current device where the data was produced, and the CUDA stream the data was produced with
* [CUDA event for synchronization between multiple CUDA streams](#synchronizing-between-cuda-streams)

Note that the `CUDA<T>` wrapper can be constructed only with
`CUDAScopedContext::wrap()`, and the data `T` can be obtained from it
only with `CUDAScopedContext::get()`, as described further below.

## CUDA EDProducer

### Class declaration

For time being (may disappear in the future) a CUDA producer should
inherit from `CUDAStreamEDProducer<...>`. The template parameters are
the usual
[stream producer extensions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface#Template_Arguments).
Note that contrary to `HeterogeneousEDProducer`, the `ExternalWork`
extension is **not** implied.

```cpp
#include "HeterogeneousCore/CUDACore/interface/CUDAStreamEDProducer.h"
class FooProducerCUDA: public CUDAStreamEDProducer<> {
  ...
```

### Memory allocation

The only effect of the `CUDAStreamEDProducer` base class is that
`beginStream(edm::StreamID)` is replaced with
`beginStreamCUDA(edm::StreamID)`. This is done in order to set the
current CUDA device before the user code starts. **If the algorithm
has to allocate memory buffers for the duration of the whole job, the
recommended place is here.** Note that a CUDA stream is not passed to
the user code. If a CUDA stream is really needed, the developer should
create+synchronize it by him/herself. (although if this appears to be
common practice, we should try to provide the situation somehow)

### Setting the current device

A CUDA producer should read either `CUDAToken` (from
`CUDADeviceChooser`) or one or more `CUDA<T>` products. Then, in the
`acquire()`/`produce()`, it should construct `CUDAScopedContext` from
one of them
```cpp
// From CUDAToken
edm::Handle<CUDAToken> htoken;
iEvent.getByToken(srcToken_, htoken);
auto ctx = CUDAScopedContext(*htoken);

/// From CUDA<T>
edm::Handle<CUDA<GPUClusters> > handle;
iEvent.getByToken(srctoken_, handle);
auto ctx = CUDAScopedContext(*handle);
```

`CUDAScopedContext` works in the RAII way and does the following
* Sets the current device (for the scope) from `CUDAToken`/`CUDA<T>`
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
edm::Handle<CUDA<GPUClusters> > hclus;
iEvent.getByToken(srctoken_, hclus);
GPUClusters const& clus = ctx.get(*hclus);
```

This step is needed to
* check that the data are on the same CUDA device
  * if not, throw an exception (with unified memory could prefetch instead)
* if the CUDA streams are different, synchronize between them

### Calling the CUDA kernels

There is nothing special, except the CUDA stream can be obtained from
the `CUDAScopedContext`

```cpp
gpuAlgo.makeClustersAsync(..., ctx.stream());
```

### Putting output

The GPU data needs to be wrapped to `CUDA<T>` template with `CUDAScopedContest.wrap()`

```cpp
GPUClusters clusters = gpuAlgo.makeClustersAsync(..., ctx.stream());
std::unique_ptr<CUDA<GPUClusters> > ret = ctx.wrap(clusters);
iEvent.put(std::move(ret));

// or with one line
iEvent.put(ctx.wrap(gpuAlgo.makeClustersAsync(ctx.stream())));
```

This step is needed to
* store the current device and CUDA stream into `CUDA<T>`
* record the CUDA event needed for CUDA stream synchronization

### `ExternalWork` extension

Everything above works both with and without `ExternalWork`.

Without `ExternalWork` the `EDProducer`s act similar to TBB
flowgraph's "streaming node". I.e. they just queue more asynchronous
work in their `produce()`.

The `ExternalWork` is needed when one would otherwise call
`cudeStreamSynchronize()`, e.g. transferring something to CPU needed
for downstream DQM, or to queue more asynchronous work. With
`ExternalWork` an `acquire()` method needs to be implemented that gets
an `edm::WaitingTaskWithArenaHolder` parameter. The
`WaitingTaskWithArenaHolder` should then be passed to the constructor
of `CUDAScopedContext` along

```cpp
void acquire(..., edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  edm::Handle<CUDA<GPUClusters> > handle;
  iEvent.getByToken(token_, handle);
  auto ctx = CUDAScopedContext(*handle, std::move(waitingTaskHolder)); // can also copy instead of move if waitingTaskHolder is needed for something else as well
  ...
```

When constructed this way, `CUDAScopedContext` registers a callback
function to the CUDA stream in its destructor to call
`waitingTaskHolder.doneWaiting()`.

A GPU->GPU producer needs a `CUDAScopedContext` also in its
`produce()`. Currently the best way is to read the input again in
`produce()` and construct the `CUDAScopedContext` from there. This
point will be improved. 

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
  * Note: `CUDAScopedContext` is **not** needed in in `produce()`

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

```python
process.fooCPU = cms.EDProducer("FooProducer") # legacy CPU

process.fooCUDADevice = cms.EDProducer("CUDADeviceChooser")
process.fooCUDADeviceFilter = cms.EDFilter("CUDADeviceFilter",
    src = cms.InputTag("fooCUDADevice")
)
process.fooCUDA = cms.EDProducer("FooProducerCUDA")
process.fooFromCUDA = cms.EDProducer("FooProducerCUDAtoCPU", src="fooCUDA")
process.foo = cms.EDProducer("FooProducerFallback",
    src = cms.VInputTag("fooFromCUDA", "fooCPU")
)
process.fooPathCUDA = cms.Path(
    process.fooCUDADeviceFilter + process.fooCUDA
)
process.fooPathCPU = cms.Path(
    ~process.fooCUDADeviceFilter + process.fooCPU
)
process.fooTask = cms.Task(
    process.fooDevice,
    process.fooFromCUDA,
    process.foo
)
...
```
For a more complete example, see [here](test/testCUDA_cfg.py).

# Extension to other devices

The C++ side extends in a straightforward way. One has to add classes
similar to `CUDAToken`, `CUDA<T>`, and `CUDAScopedContext`. Of course,
much depends on the exact details. The python configuration side
extends as well, one "just" has to add more modules there. Also the
device choosing logic is also extendable
```python
process.fooCUDADevice = ...
process.fooFPGADevice = ...
process.fooPathCUDA = cms.Path(
    process.fooCUDADeviceFilter + ...
)
process.fooPathFPGA = cms.Path(
    ~process.fooCUDADeviceFilter + process.fooFPGADeviceFilter + ...
)    
process.fooPathCPU = cms.Path(
    ~process.fooCUDADeviceFilter + ~process.fooFPGADeviceFilter + ...
)
```
