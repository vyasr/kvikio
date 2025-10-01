/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <utility>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/nvtx.hpp>
#include "cuda.h"

namespace kvikio {

FileHandle::FileHandle(std::string const& file_path,
                       std::string const& flags,
                       mode_t mode,
                       CompatMode compat_mode)
  : _initialized{true}, _compat_mode_manager{file_path, flags, mode, compat_mode, this}
{
  KVIKIO_NVTX_FUNC_RANGE();
}

FileHandle::FileHandle(FileHandle&& o) noexcept
  : _file_direct_on{std::exchange(o._file_direct_on, {})},
    _file_direct_off{std::exchange(o._file_direct_off, {})},
    _initialized{std::exchange(o._initialized, false)},
    _nbytes{std::exchange(o._nbytes, 0)},
    _cufile_handle{std::exchange(o._cufile_handle, {})},
    _compat_mode_manager{std::move(o._compat_mode_manager)}
{
}

FileHandle& FileHandle::operator=(FileHandle&& o) noexcept
{
  _file_direct_on      = std::exchange(o._file_direct_on, {});
  _file_direct_off     = std::exchange(o._file_direct_off, {});
  _initialized         = std::exchange(o._initialized, false);
  _nbytes              = std::exchange(o._nbytes, 0);
  _cufile_handle       = std::exchange(o._cufile_handle, {});
  _compat_mode_manager = std::move(o._compat_mode_manager);
  return *this;
}

FileHandle::~FileHandle() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  close();
}

bool FileHandle::closed() const noexcept { return !_initialized; }

void FileHandle::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  try {
    if (closed()) { return; }
    _cufile_handle.unregister_handle();
    _file_direct_off.close();
    _file_direct_on.close();
    _nbytes      = 0;
    _initialized = false;
  } catch (...) {
  }
}

CUfileHandle_t FileHandle::handle()
{
  KVIKIO_EXPECT(!closed(), "File handle is closed");
  KVIKIO_EXPECT(!get_compat_mode_manager().is_compat_mode_preferred(),
                "The underlying cuFile handle isn't available in compatibility mode");
  return _cufile_handle.handle();
}

int FileHandle::fd(bool o_direct) const noexcept
{
  return o_direct ? _file_direct_on.fd() : _file_direct_off.fd();
}

int FileHandle::fd_open_flags(bool o_direct) const { return open_flags(fd(o_direct)); }

std::size_t FileHandle::nbytes() const
{
  if (closed()) { return 0; }
  if (_nbytes == 0) { _nbytes = get_file_size(_file_direct_off.fd()); }
  return _nbytes;
}

std::size_t FileHandle::read(void* devPtr_base,
                             std::size_t size,
                             std::size_t file_offset,
                             std::size_t devPtr_offset,
                             bool sync_default_stream)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  if (get_compat_mode_manager().is_compat_mode_preferred()) {
    return detail::posix_device_read(
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

  ssize_t ret = cuFileAPI::instance().Read(_cufile_handle.handle(),
                                           devPtr_base,
                                           size,
                                           convert_size2off(file_offset),
                                           convert_size2off(devPtr_offset));
  CUFILE_CHECK_BYTES_DONE(ret);
  return ret;
}

std::size_t FileHandle::write(void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              bool sync_default_stream)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  _nbytes = 0;  // Invalidate the computed file size

  if (get_compat_mode_manager().is_compat_mode_preferred()) {
    return detail::posix_device_write(
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

  ssize_t ret = cuFileAPI::instance().Write(_cufile_handle.handle(),
                                            devPtr_base,
                                            size,
                                            convert_size2off(file_offset),
                                            convert_size2off(devPtr_offset));
  KVIKIO_EXPECT(ret != -1, "Unable to write file", GenericSystemError);
  KVIKIO_EXPECT(ret >= 0, std::string{"cuFile error:"} + CUFILE_ERRSTR(ret));
  return ret;
}

std::future<std::size_t> FileHandle::pread(void* buf,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           std::size_t task_size,
                                           std::size_t gds_threshold,
                                           bool sync_default_stream)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);
  if (is_host_memory(buf)) {
    auto op = [this](void* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char* buf = static_cast<char*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_read<detail::PartialIO::NO>(
        _file_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0, call_idx, nvtx_color);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_read = detail::posix_device_read(_file_direct_off.fd(), buf, size, file_offset, 0);
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_read);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !get_compat_mode_manager().is_compat_mode_preferred()) {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
  }

  // Regular case that use the threadpool and run the tasks in parallel
  auto task = [this, ctx](void* devPtr_base,
                          std::size_t size,
                          std::size_t file_offset,
                          std::size_t devPtr_offset) -> std::size_t {
    PushAndPopContext c(ctx);
    return read(devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */ false);
  };
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
  return parallel_io(
    task, devPtr_base, size, file_offset, task_size, devPtr_offset, call_idx, nvtx_color);
}

std::future<std::size_t> FileHandle::pwrite(void const* buf,
                                            std::size_t size,
                                            std::size_t file_offset,
                                            std::size_t task_size,
                                            std::size_t gds_threshold,
                                            bool sync_default_stream)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);
  if (is_host_memory(buf)) {
    auto op = [this](void const* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char const* buf = static_cast<char const*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_write<detail::PartialIO::NO>(
        _file_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0, call_idx, nvtx_color);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_write = detail::posix_device_write(_file_direct_off.fd(), buf, size, file_offset, 0);
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_write);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !get_compat_mode_manager().is_compat_mode_preferred()) {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
  }

  // Regular case that use the threadpool and run the tasks in parallel
  auto op = [this, ctx](void const* devPtr_base,
                        std::size_t size,
                        std::size_t file_offset,
                        std::size_t devPtr_offset) -> std::size_t {
    PushAndPopContext c(ctx);
    return write(devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */ false);
  };
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
  return parallel_io(
    op, devPtr_base, size, file_offset, task_size, devPtr_offset, call_idx, nvtx_color);
}

// TODO: This is a free function but it is implicitly tied to the lifetime of
// the calling FileHandle. Is there a way that we can make that dependence
// manifest?
void do_pread_async(int fd,
                    void* devPtr_base,
                    std::size_t* size_p,
                    off_t* file_offset_p,
                    off_t* devPtr_offset_p,
                    ssize_t* bytes_read_p,
                    std::size_t task_size,
                    std::size_t gds_threshold,
                    bool compat_mode_preferred_for_async,
                    CUFileHandleWrapper& cufile_handle,
                    CUstream stream,
                    CUevent memory_ready,
                    CUevent pread_async_done)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(call_idx, nvtx_color);

  // Wait on the memory_ready event to ensure that upstream work is done and
  // all the pointer inputs have been populated.
  auto ctx = get_context_from_pointer(devPtr_base);
  {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().EventSynchronize(memory_ready));
  }

  if (*size_p < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_read = detail::posix_device_read(
      fd, devPtr_base, *size_p, *file_offset_p, *devPtr_offset_p, detail::StreamsByThread::get());
  } else if (compat_mode_preferred_for_async) {
    // TODO: We could include an extra case for CUDA 12.0-12.2 where we have
    // cufile but not the async APIs and fall back to those. I think that
    // case is too narrow to be worth handling, though, and also we'd be at
    // the mercy of cufile's internal stream usage to avoid surprising syncs,
    // so I would rather we just do the predictable POSIX read in that case.
    auto task = [=](void* devPtr_base,
                    std::size_t size,
                    off_t file_offset_p,
                    off_t devPtr_offset_p) -> std::size_t {
      CUcontext ctx = get_context_from_pointer(devPtr_base);
      PushAndPopContext c(ctx);
      return detail::posix_device_read(fd, devPtr_base, size, file_offset_p, devPtr_offset_p);
    };

    // Now launch all the tasks
    // TODO: Is it worth reducing some overhead by writing a version of this
    // code that doesn't produce a future? Probably not since we need all but the final task
    // to still use futures.
    auto ret = parallel_io(task, devPtr_base, *size_p, *file_offset_p, task_size, *devPtr_offset_p);
    *bytes_read_p = ret.get();
  } else {
    // In this case we use a single cuFileReadAsync call
    // TODO: Test the behavior of submitting multiple cuFileReadAsync calls in a thread pool as
    // well.
    // TODO: Figure out how to validate that async compat mode is allowed
    // get_compat_mode_manager().validate_compat_mode_for_async();
    CUcontext ctx = get_context_from_pointer(devPtr_base);
    PushAndPopContext c(ctx);
    cuFileAPI::instance().ReadAsync(cufile_handle.handle(),
                                    devPtr_base,
                                    size_p,
                                    file_offset_p,
                                    devPtr_offset_p,
                                    bytes_read_p,
                                    stream);
  }
  // In the case where we used cuFileReadAsync, we don't actually need the
  // event since the operation is intrinsically stream-ordered, but we need it
  // in all other cases so we must do it unconditionally so that we can wait on
  // the event in the calling thread.
  CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(pread_async_done, stream));
  return;
}

/*
 * Function flow:
 * 1. Create and record event A on the input stream
 * 2. Create event B
 * 3. Launch a new thread (call it the worker thread), pass it event B
 * 4. The worker thread waits on event A with cudaEventSynchronize
 * 5. The worker thread checks the input size.
 * 6. If the size is below the threshold:
 *     1. Do a single POSIX read
 *     2. cudaMemcpyAsync on the original input stream
 *     3. Synchronizes the stream (necessary to ensure the lifetime of the POSIX-read buffer lasts
 * through the cudaMemcpyAsync)
 *     4. cudaEventRecord of event B to signal that the work is done
 * 7. If the size is above the threshold:
 *     1. If GDS is available
 *         1. Use cuFileReadAsync
 *         2. cudaEventRecord of event B to signal that the work is done
 *     2. If GDS is not available:
 *         1. Create a thread pool
 *         2. Split up the work into multiple tasks based on the size
 *         3. Each task does a POSIX read followed by a cudaMemcpyAsync on a StreamPerThread stream
 *         4. The worker thread does the final read and then waits on all of the tasks to finish
 *         5. cudaEventRecord of event B to signal that the work is done
 */

void FileHandle::pread_async(void* devPtr_base,
                             std::size_t* size_p,
                             off_t* file_offset_p,
                             off_t* devPtr_offset_p,
                             ssize_t* bytes_read_p,
                             std::size_t task_size,
                             std::size_t gds_threshold,
                             CUstream stream)
{
  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(call_idx, nvtx_color);

  KVIKIO_EXPECT(!is_host_memory(devPtr_base), "devPtr_base must be a device pointer");
  CUcontext ctx = get_context_from_pointer(devPtr_base);
  KVIKIO_EXPECT(
    stream == nullptr || get_context_associated_pointer(convert_void2deviceptr(stream)).has_value(),
    "stream must be associated with the context");

  // memory_ready is the indicator that upstream work is completed, so we
  // record it immediately and send it to the worker threads to wait on.
  // pread_async_done is the indication that the worker threads are completed
  // and therefore what the main thread will tell the calling stream to wait
  // on at the end of this function
  CUevent memory_ready, pread_async_done;
  {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&memory_ready, 0));
    CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(memory_ready, stream));

    CUDA_DRIVER_TRY(cudaAPI::instance().EventCreate(&pread_async_done, 0));
  }

  // All synchronization is done via streams, so we use detach_task instead of submit_task
  defaults::thread_pool().detach_task([=, this] {
    KVIKIO_NVTX_SCOPED_RANGE("pread_async_task", call_idx, nvtx_color);
    return do_pread_async(this->_file_direct_off.fd(),
                          devPtr_base,
                          size_p,
                          file_offset_p,
                          devPtr_offset_p,
                          bytes_read_p,
                          task_size,
                          gds_threshold,
                          get_compat_mode_manager().is_compat_mode_preferred_for_async(),
                          this->_cufile_handle,
                          stream,
                          memory_ready,
                          pread_async_done);
  });
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamWaitEvent(stream, pread_async_done, 0));
}

void FileHandle::read_async(void* devPtr_base,
                            std::size_t* size_p,
                            off_t* file_offset_p,
                            off_t* devPtr_offset_p,
                            ssize_t* bytes_read_p,
                            CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  get_compat_mode_manager().validate_compat_mode_for_async();
  if (get_compat_mode_manager().is_compat_mode_preferred_for_async()) {
    *bytes_read_p = static_cast<ssize_t>(detail::posix_device_read(
      _file_direct_off.fd(), devPtr_base, *size_p, *file_offset_p, *devPtr_offset_p, stream));
  } else {
    CUFILE_TRY(cuFileAPI::instance().ReadAsync(_cufile_handle.handle(),
                                               devPtr_base,
                                               size_p,
                                               file_offset_p,
                                               devPtr_offset_p,
                                               bytes_read_p,
                                               stream));
  }
}

StreamFuture FileHandle::read_async(
  void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
  auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_] =
    ret.get_args();
  read_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_);
  return ret;
}

void FileHandle::write_async(void* devPtr_base,
                             std::size_t* size_p,
                             off_t* file_offset_p,
                             off_t* devPtr_offset_p,
                             ssize_t* bytes_written_p,
                             CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  get_compat_mode_manager().validate_compat_mode_for_async();
  if (get_compat_mode_manager().is_compat_mode_preferred_for_async()) {
    *bytes_written_p = static_cast<ssize_t>(detail::posix_device_write(
      _file_direct_off.fd(), devPtr_base, *size_p, *file_offset_p, *devPtr_offset_p, stream));
  } else {
    CUFILE_TRY(cuFileAPI::instance().WriteAsync(_cufile_handle.handle(),
                                                devPtr_base,
                                                size_p,
                                                file_offset_p,
                                                devPtr_offset_p,
                                                bytes_written_p,
                                                stream));
  }
}

StreamFuture FileHandle::write_async(
  void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
  auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_] =
    ret.get_args();
  write_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_);
  return ret;
}

const CompatModeManager& FileHandle::get_compat_mode_manager() const noexcept
{
  return _compat_mode_manager;
}

}  // namespace kvikio
