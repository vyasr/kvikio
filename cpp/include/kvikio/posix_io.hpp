/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once

#include <unistd.h>
#include <cstddef>
#include <cstdlib>

#include <cuda.h>

#include <cstring>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace {

inline constexpr std::size_t page_size  = 2 << 9;   // 4 KiB
inline constexpr std::size_t chunk_size = 2 << 23;  // 16 MiB

/**
 * @brief Call ::pwrite() until all of `count` has been written
 *
 * @param fd File decriptor
 * @param buf Buffer to write
 * @param count Number of bytes to write
 * @param offset File offset
 */
inline void pwrite_all(int fd, const void* buf, size_t count, off_t offset)
{
  off_t cur_offset      = offset;
  size_t byte_remaining = count;
  const char* buffer    = static_cast<const char*>(buf);
  while (byte_remaining > 0) {
    ssize_t nbytes_written = ::pwrite(fd, buffer, byte_remaining, cur_offset);
    if (nbytes_written == -1) {
      if (errno == EBADF) {
        throw CUfileException{std::string{"POSIX error on pread at: "} + __FILE__ + ":" +
                              KVIKIO_STRINGIFY(__LINE__) + ": unsupported file open flags"};
      }
      throw CUfileException{std::string{"POSIX error on pwrite at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + strerror(errno)};
    }
    if (nbytes_written == 0) {
      throw CUfileException{std::string{"POSIX error on pwrite at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": EOF"};
    }

    buffer += nbytes_written;
    cur_offset += nbytes_written;
    byte_remaining -= nbytes_written;
  }
}

/**
 * @brief Read or write main memory to or from disk using POSIX
 *
 * @tparam IsReadOperation Whether the operation is a read or a write
 * @param fd File decriptor
 * @param devPtr_base Device pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param devPtr_offset Byte offset to the start of the device pointer.
 * @return Number of bytes read or written.
 */
template <bool IsReadOperation>
inline std::size_t posix_io(int fd,
                            const void* devPtr_base,
                            std::size_t size,
                            std::size_t file_offset,
                            std::size_t devPtr_offset)
{
  void* buf = nullptr;
  {
    // TODO: reuse memory allocations
    int err = ::posix_memalign(&buf, page_size, chunk_size);
    if (err != 0) {
      throw CUfileException{std::string{"POSIX error at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + strerror(err)};
    }
  }
  try {
    CUdeviceptr devPtr      = convert_void2deviceptr(devPtr_base) + devPtr_offset;
    off_t cur_file_offset   = convert_size2off(file_offset);
    off_t byte_remaining    = convert_size2off(size);
    const off_t chunk_size2 = convert_size2off(chunk_size);

    while (byte_remaining > 0) {
      const off_t nbytes_requested = std::min(chunk_size2, byte_remaining);
      ssize_t nbytes_got           = nbytes_requested;
      if constexpr (IsReadOperation) {
        nbytes_got = ::pread(fd, buf, nbytes_requested, cur_file_offset);
        if (nbytes_got == -1) {
          if (errno == EBADF) {
            throw CUfileException{std::string{"POSIX error on pread at: "} + __FILE__ + ":" +
                                  KVIKIO_STRINGIFY(__LINE__) + ": unsupported file open flags"};
          }
          throw CUfileException{std::string{"POSIX error on pread at: "} + __FILE__ + ":" +
                                KVIKIO_STRINGIFY(__LINE__) + ": " + strerror(errno)};
        }
        if (nbytes_got == 0) {
          throw CUfileException{std::string{"POSIX error on pread at: "} + __FILE__ + ":" +
                                KVIKIO_STRINGIFY(__LINE__) + ": EOF"};
        }
        CUDA_DRIVER_TRY(cuMemcpyHtoD(devPtr, buf, nbytes_got));
      } else {  // Is a write operation
        CUDA_DRIVER_TRY(cuMemcpyDtoH(buf, devPtr, nbytes_requested));
        pwrite_all(fd, buf, nbytes_requested, cur_file_offset);
      }
      cur_file_offset += nbytes_got;
      devPtr += nbytes_got;
      byte_remaining -= nbytes_got;
    }
  } catch (...) {
    free(buf);
    throw;
  }
  return size;
}

}  // namespace

/**
 * @brief Read main memory from disk using POSIX
 *
 * @param fd File decriptor
 * @param devPtr_base Base address of buffer in device memory.
 * @param size Size in bytes to read.
 * @param file_offset Offset in the file to read from.
 * @param devPtr_offset Offset relative to the `devPtr_base` pointer to read into.
 * @return Size of bytes that were successfully read.
 */
inline std::size_t posix_read(int fd,
                              const void* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset)
{
  return posix_io<true>(fd, devPtr_base, size, file_offset, devPtr_offset);
}

/**
 * @brief Write main memory to disk using POSIX
 *
 * @param fd File decriptor
 * @param devPtr_base Base address of buffer in device memory.
 * @param size Size in bytes to write.
 * @param file_offset Offset in the file to write from.
 * @param devPtr_offset Offset relative to the `devPtr_base` pointer to write into.
 * @return Size of bytes that were successfully written.
 */
inline std::size_t posix_write(int fd,
                               const void* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset)
{
  return posix_io<false>(fd, devPtr_base, size, file_offset, devPtr_offset);
}

}  // namespace kvikio
