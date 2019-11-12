
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

//#include "dispatch/dispatch_scan.cuh"
#include "cub/agent/agent_scan.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/grid/grid_queue.cuh"
#include "cub/util_arch.cuh"
#include "cub/util_debug.cuh"

//#include "cub/util_device.cuh"
#include "cub/util_type.cuh"
#include "cub/util_macro.cuh"

#include "cub/util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace notcub {
/**
 * Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t  &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return CubDebug(cudaErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *) ((size_t(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return cudaSuccess;
}


/**
 * Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }


/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t PtxVersion(int &ptx_version)
{
    struct Dummy
    {
        /// Type definition of the EmptyKernel kernel entry point
        typedef void (*EmptyKernelPtr)();

        /// Force EmptyKernel<void> to be generated if this class is used
        CUB_RUNTIME_FUNCTION __forceinline__
        EmptyKernelPtr Empty()
        {
            return EmptyKernel<void>;
        }
    };

    ptx_version = 600;
    return cudaSuccess;

    /*
#ifndef CUB_RUNTIME_ENABLED
    (void)ptx_version;

    // CUDA API calls not supported from this device
    return cudaErrorInvalidConfiguration;

#elif (CUB_PTX_ARCH > 0)

    ptx_version = CUB_PTX_ARCH;
    return cudaSuccess;

#else

    cudaError_t error = cudaSuccess;
    do
    {
        cudaFuncAttributes empty_kernel_attrs;
        if (CubDebug(error = cudaFuncGetAttributes(&empty_kernel_attrs, EmptyKernel<void>))) break;
        ptx_version = empty_kernel_attrs.ptxVersion * 10;
    }
    while (0);

    return error;

#endif
    */
}


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Synchronize the stream if specified
 */
CUB_RUNTIME_FUNCTION __forceinline__
static cudaError_t SyncStream(cudaStream_t stream)
{
#if (CUB_PTX_ARCH == 0)
    return cudaStreamSynchronize(stream);
#else
    (void)stream;
    // Device can't yet sync on a specific stream
    return cudaDeviceSynchronize();
#endif
}


/**
 * \brief Computes maximum SM occupancy in thread blocks for executing the given kernel function pointer \p kernel_ptr on the current device with \p block_threads per thread block.
 *
 * \par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * \endcode
 *
 */
template <typename KernelPtr>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t MaxSmOccupancy(
    int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads,              ///< [in] Number of threads per thread block
    int                 dynamic_smem_bytes = 0)
{
#ifndef CUB_RUNTIME_ENABLED
    (void)dynamic_smem_bytes;
    (void)block_threads;
    (void)kernel_ptr;
    (void)max_sm_occupancy;

    // CUDA API calls not supported from this device
    return CubDebug(cudaErrorInvalidConfiguration);

#else

    return cudaOccupancyMaxActiveBlocksPerMultiprocessor (
        &max_sm_occupancy,
        kernel_ptr,
        block_threads,
        dynamic_smem_bytes);

#endif  // CUB_RUNTIME_ENABLED
}


/******************************************************************************
 * Policy management
 ******************************************************************************/

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_size;
    int sm_occupancy;

    CUB_RUNTIME_FUNCTION __forceinline__
    KernelConfig() : block_threads(0), items_per_thread(0), tile_size(0), sm_occupancy(0) {}

    template <typename AgentPolicyT, typename KernelPtrT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Init(KernelPtrT kernel_ptr)
    {
        block_threads        = AgentPolicyT::BLOCK_THREADS;
        items_per_thread     = AgentPolicyT::ITEMS_PER_THREAD;
        tile_size            = block_threads * items_per_thread;
        cudaError_t retval   = notcub::MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
        return retval;
    }
};



/// Helper for dispatching into a policy chain
template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
   /// The policy for the active compiler pass
  typedef typename cub::If<(CUB_PTX_ARCH < PTX_VERSION), typename PrevPolicyT::ActivePolicy, PolicyT>::Type ActivePolicy;

   /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
   template <typename FunctorT>
   CUB_RUNTIME_FUNCTION __forceinline__
   static cudaError_t Invoke(int ptx_version, FunctorT &op)
   {
       if (ptx_version < PTX_VERSION) {
           return PrevPolicyT::Invoke(ptx_version, op);
       }
       return op.template Invoke<PolicyT>();
   }
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PTX_VERSION, typename PolicyT>
struct ChainedPolicy<PTX_VERSION, PolicyT, PolicyT>
{
    /// The policy for the active compiler pass
    typedef PolicyT ActivePolicy;

    /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
    template <typename FunctorT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Invoke(int /*ptx_version*/, FunctorT &op) {
        return op.template Invoke<PolicyT>();
    }
};




#endif  // Do not document


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename            ScanTileStateT>     ///< Tile status interface type
__global__ void DeviceScanInitKernel(
    ScanTileStateT      tile_state,         ///< [in] Tile status interface
    int                 num_tiles)          ///< [in] Number of tiles
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename                ScanTileStateT,         ///< Tile status interface type
    typename                NumSelectedIteratorT>   ///< Output iterator type for recording the number of items selected
__global__ void DeviceCompactInitKernel(
    ScanTileStateT          tile_state,             ///< [in] Tile status interface
    int                     num_tiles,              ///< [in] Number of tiles
    NumSelectedIteratorT    d_num_selected_out)     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);

    // Initialize d_num_selected_out
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
        *d_num_selected_out = 0;
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename            ScanPolicyT,        ///< Parameterized ScanPolicyT tuning policy type
    typename            InputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename            OutputIteratorT,    ///< Random-access output iterator type for writing scan outputs \iterator
    typename            ScanTileStateT,     ///< Tile status interface type
    typename            ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            InitValueT,         ///< Initial value to seed the exclusive scan (cub::NullType for inclusive scans)
    typename            OffsetT>            ///< Signed integer type for global offsets
__launch_bounds__ (int(ScanPolicyT::BLOCK_THREADS))
__global__ void DeviceScanKernel(
    InputIteratorT      d_in,               ///< Input data
    OutputIteratorT     d_out,              ///< Output data
    ScanTileStateT      tile_state,         ///< Tile status interface
    int                 start_tile,         ///< The starting tile for the current grid
    ScanOpT             scan_op,            ///< Binary scan functor 
    InitValueT          init_value,         ///< Initial value to seed the exclusive scan
    OffsetT             num_items)          ///< Total number of scan items for the entire problem
{
    // Thread block type for scanning input tiles
  typedef cub::AgentScan<
        ScanPolicyT,
        InputIteratorT,
        OutputIteratorT,
        ScanOpT,
        InitValueT,
        OffsetT> AgentScanT;

    // Shared memory for AgentScan
    __shared__ typename AgentScanT::TempStorage temp_storage;

    // Process tiles
    AgentScanT(temp_storage, d_in, d_out, scan_op, init_value).ConsumeRange(
        num_items,
        tile_state,
        start_tile);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/


/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename InputIteratorT,     ///< Random-access input iterator type for reading scan inputs \iterator
    typename OutputIteratorT,    ///< Random-access output iterator type for writing scan outputs \iterator
    typename ScanOpT,            ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename InitValueT,          ///< The init_value element type for ScanOpT (cub::NullType for inclusive scans)
    typename OffsetT>            ///< Signed integer type for global offsets
struct DispatchScan
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The output value type
  typedef typename cub::If<(cub::Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
        typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
        typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

    // Tile status descriptor interface type
  typedef cub::ScanTileState<OutputT> ScanTileStateT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    /// SM600
    struct Policy600
    {
      typedef cub::AgentScanPolicy<
            CUB_SCALED_GRANULARITIES(128, 15, OutputT),      ///< Threads per block, items per thread
                cub::BLOCK_LOAD_TRANSPOSE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_STORE_TRANSPOSE,
                cub::BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    /// SM520
    struct Policy520
    {
        // Titan X: 32.47B items/s @ 48M 32-bit T
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, OutputT),      ///< Threads per block, items per thread
                cub::BLOCK_LOAD_DIRECT,
                cub::LOAD_LDG,
                cub::BLOCK_STORE_WARP_TRANSPOSE,
                cub::BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    /// SM35
    struct Policy350
    {
        // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, OutputT),      ///< Threads per block, items per thread
                cub::BLOCK_LOAD_DIRECT,
                cub::LOAD_LDG,
                cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
        cub::BLOCK_SCAN_RAKING>
            ScanPolicyT;
    };

    /// SM30
    struct Policy300
    {
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(256, 9, OutputT),      ///< Threads per block, items per thread
                cub::BLOCK_LOAD_WARP_TRANSPOSE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_STORE_WARP_TRANSPOSE,
                cub::BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };

    /// SM20
    struct Policy200
    {
        // GTX 580: 20.3B items/s (162.3 GB/s) @ 48M 32-bit T
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(128, 12, OutputT),      ///< Threads per block, items per thread
        cub::BLOCK_LOAD_WARP_TRANSPOSE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_STORE_WARP_TRANSPOSE,
                cub::BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };

    /// SM13
    struct Policy130
    {
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(96, 21, OutputT),      ///< Threads per block, items per thread
        cub::BLOCK_LOAD_WARP_TRANSPOSE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_STORE_WARP_TRANSPOSE,
        cub::BLOCK_SCAN_RAKING_MEMOIZE>
            ScanPolicyT;
    };

    /// SM10
    struct Policy100
    {
      typedef cub::AgentScanPolicy<
                CUB_SCALED_GRANULARITIES(64, 9, OutputT),      ///< Threads per block, items per thread
        cub::BLOCK_LOAD_WARP_TRANSPOSE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_STORE_WARP_TRANSPOSE,
                cub::BLOCK_SCAN_WARP_SCANS>
            ScanPolicyT;
    };


    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 600)
    typedef Policy600 PtxPolicy;

#elif (CUB_PTX_ARCH >= 520)
    typedef Policy520 PtxPolicy;

#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_ARCH >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxAgentScanPolicy : PtxPolicy::ScanPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &scan_kernel_config)
    {
    #if (CUB_PTX_ARCH > 0)
        (void)ptx_version;

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        scan_kernel_config.template Init<PtxAgentScanPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 600)
        {
            scan_kernel_config.template Init<typename Policy600::ScanPolicyT>();
        }
        else if (ptx_version >= 520)
        {
            scan_kernel_config.template Init<typename Policy520::ScanPolicyT>();
        }
        else if (ptx_version >= 350)
        {
            scan_kernel_config.template Init<typename Policy350::ScanPolicyT>();
        }
        else if (ptx_version >= 300)
        {
            scan_kernel_config.template Init<typename Policy300::ScanPolicyT>();
        }
        else if (ptx_version >= 200)
        {
            scan_kernel_config.template Init<typename Policy200::ScanPolicyT>();
        }
        else if (ptx_version >= 130)
        {
            scan_kernel_config.template Init<typename Policy130::ScanPolicyT>();
        }
        else
        {
            scan_kernel_config.template Init<typename Policy100::ScanPolicyT>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide prefix scan using the
     * specified kernel functions.
     */
    template <
        typename            ScanInitKernelPtrT,     ///< Function type of cub::DeviceScanInitKernel
        typename            ScanSweepKernelPtrT>    ///< Function type of cub::DeviceScanKernelPtrT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*               d_temp_storage,         ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                   ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                  ///< [out] Pointer to the output sequence of data items
        ScanOpT             scan_op,                ///< [in] Binary scan functor 
        InitValueT          init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT             num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t        stream,                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous,      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                 /*ptx_version*/,        ///< [in] PTX version of dispatch kernels
        ScanInitKernelPtrT  init_kernel,            ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
        ScanSweepKernelPtrT scan_kernel,            ///< [in] Kernel function pointer to parameterization of cub::DeviceScanKernel
        KernelConfig        scan_kernel_config)     ///< [in] Dispatch parameters that match the policy that \p scan_kernel was compiled for
    {

#ifndef CUB_RUNTIME_ENABLED
        (void)d_temp_storage;
        (void)temp_storage_bytes;
        (void)d_in;
        (void)d_out;
        (void)scan_op;
        (void)init_value;
        (void)num_items;
        (void)stream;
        (void)debug_synchronous;
        (void)init_kernel;
        (void)scan_kernel;
        (void)scan_kernel_config;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Number of input tiles
            int tile_size = scan_kernel_config.block_threads * scan_kernel_config.items_per_thread;
            int num_tiles = (num_items + tile_size - 1) / tile_size;

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[1];
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void* allocations[1];
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Return if empty problem
            if (num_items == 0)
                break;

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS;
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            init_kernel<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(
                tile_state,
                num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Get SM occupancy for scan_kernel
            int scan_sm_occupancy;
            if (CubDebug(error = notcub::MaxSmOccupancy(
                scan_sm_occupancy,            // out
                scan_kernel,
                scan_kernel_config.block_threads))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Run grids in epochs (in case number of tiles exceeds max x-dimension
            int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
            for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
            {
                // Log scan_kernel configuration
                if (debug_synchronous) _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    start_tile, scan_grid_size, scan_kernel_config.block_threads, (long long) stream, scan_kernel_config.items_per_thread, scan_sm_occupancy);

                // Invoke scan_kernel
                scan_kernel<<<scan_grid_size, scan_kernel_config.block_threads, 0, stream>>>(
                    d_in,
                    d_out,
                    tile_state,
                    start_tile,
                    scan_op,
                    init_value,
                    num_items);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*           d_temp_storage,         ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&         temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                   ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                  ///< [out] Pointer to the output sequence of data items
        ScanOpT         scan_op,                ///< [in] Binary scan functor 
        InitValueT      init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT         num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Get kernel kernel dispatch configurations
            KernelConfig scan_kernel_config;
            InitConfigs(ptx_version, scan_kernel_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                scan_op,
                init_value,
                num_items,
                stream,
                debug_synchronous,
                ptx_version,
                DeviceScanInitKernel<ScanTileStateT>,
                DeviceScanKernel<PtxAgentScanPolicy, InputIteratorT, OutputIteratorT, ScanTileStateT, ScanOpT, InitValueT, OffsetT>,
                scan_kernel_config))) break;
        }
        while (0);

        return error;
    }
};



/**
 * \brief DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within device-accessible memory. ![](device_scan.png)
 * \ingroup SingleModule
 *
 * \par Overview
 * Given a sequence of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output sequence where each element is computed to be the reduction
 * of the elements occurring earlier in the input sequence.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par
 * As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our <em>"decoupled look-back"</em> algorithm
 * for performing global prefix scan with only a single pass through the
 * input data, as described in our 2016 technical report [1].  The central
 * idea is to leverage a small, constant factor of redundant work in order to overlap the latencies
 * of global prefix propagation with local computation.  As such, our algorithm requires only
 * ~2<em>n</em> data movement (<em>n</em> inputs are read, <em>n</em> outputs are written), and typically
 * proceeds at "memcpy" speeds.
 *
 * \par
 * [1] [Duane Merrill and Michael Garland.  "Single-pass Parallel Prefix Scan with Decoupled Look-back", <em>NVIDIA Technical Report NVR-2016-002</em>, 2016.](https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back)
 *
 * \par Usage Considerations
 * \cdp_class{DeviceScan}
 *
 * \par Performance
 * \linear_performance{prefix scan}
 *
 * \par
 * The following chart illustrates DeviceScan::ExclusiveSum
 * performance across different CUDA architectures for \p int32 keys.
 * \plots_below
 *
 * \image html scan_int32.png
 *
 */
struct DeviceScan
{
    /******************************************************************//**
     * \name Exclusive scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide exclusive prefix sum.  The value of 0 is applied as the initial value, and is assigned to *d_out.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Provides "run-to-run" determinism for pseudo-associative reduction
     *   (e.g., addition of floating point types) on the same GPU device.
     *   However, results for pseudo-associative reduction may be inconsistent
     *   from one device to a another device of a different compute-capability
     *   because CUB can employ different tile-sizing for different architectures.
     * - \devicestorage
     *
     * \par Performance
     * The following charts illustrate saturated exclusive sum performance across different
     * CUDA architectures for \p int32 and \p int64 items, respectively.
     *
     * \image html scan_int32.png
     * \image html scan_int64.png
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix sum
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveSum(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output sequence of data items
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // The output value type
        typedef typename cub::If<(cub::Equals<typename std::iterator_traits<OutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
            typename std::iterator_traits<InputIteratorT>::value_type,                                          // ... then the input iterator's value type,
            typename std::iterator_traits<OutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

        // Initial value
        OutputT init_value = 0;

        return DispatchScan<InputIteratorT, OutputIteratorT, cub::Sum, OutputT, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            cub::Sum(),
            init_value,
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.  The \p init_value value is applied as the initial value, and is assigned to *d_out.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Provides "run-to-run" determinism for pseudo-associative reduction
     *   (e.g., addition of floating point types) on the same GPU device.
     *   However, results for pseudo-associative reduction may be inconsistent
     *   from one device to a another device of a different compute-capability
     *   because CUB can employ different tile-sizing for different architectures.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the exclusive prefix min-scan of an \p int device vector
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op
     * ...
     *
     * // Determine temporary device storage requirements for exclusive prefix scan
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // Allocate temporary storage for exclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix min-scan
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT   <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT  <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam Identity         <b>[inferred]</b> Type of the \p identity value used Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT,
        typename        InitValueT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ExclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output sequence of data items
        ScanOpT         scan_op,                            ///< [in] Binary scan functor
        InitValueT      init_value,                         ///< [in] Initial value to seed the exclusive scan (and is assigned to *d_out)
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, InitValueT, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            init_value,
            num_items,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - Provides "run-to-run" determinism for pseudo-associative reduction
     *   (e.g., addition of floating point types) on the same GPU device.
     *   However, results for pseudo-associative reduction may be inconsistent
     *   from one device to a another device of a different compute-capability
     *   because CUB can employ different tile-sizing for different architectures.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix sum
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out <-- [8, 14, 21, 26, 29, 29, 38]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     */
    template <
        typename            InputIteratorT,
        typename            OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveSum(
        void*               d_temp_storage,                 ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&             temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT      d_in,                           ///< [in] Pointer to the input sequence of data items
        OutputIteratorT     d_out,                          ///< [out] Pointer to the output sequence of data items
        int                 num_items,                      ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t        stream             = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, cub::Sum, cub::NullType, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            cub::Sum(),
            cub::NullType(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide inclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - Provides "run-to-run" determinism for pseudo-associative reduction
     *   (e.g., addition of floating point types) on the same GPU device.
     *   However, results for pseudo-associative reduction may be inconsistent
     *   from one device to a another device of a different compute-capability
     *   because CUB can employ different tile-sizing for different architectures.
     * - \devicestorage
     *
     * \par Snippet
     * The code snippet below illustrates the inclusive prefix min-scan of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device-accessible pointers for input and output
     * int          num_items;      // e.g., 7
     * int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix scan
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // Allocate temporary storage for inclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix min-scan
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // d_out <-- [8, 6, 6, 5, 3, 0, 0]
     *
     * \endcode
     *
     * \tparam InputIteratorT   <b>[inferred]</b> Random-access input iterator type for reading scan inputs \iterator
     * \tparam OutputIteratorT  <b>[inferred]</b> Random-access output iterator type for writing scan outputs \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIteratorT,
        typename        OutputIteratorT,
        typename        ScanOpT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t InclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output sequence of data items
        ScanOpT         scan_op,                            ///< [in] Binary scan functor
        int             num_items,                          ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t    stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        return DispatchScan<InputIteratorT, OutputIteratorT, ScanOpT, cub::NullType, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            cub::NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_scan.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
