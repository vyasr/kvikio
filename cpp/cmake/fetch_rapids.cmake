# =============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-21.12/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake
)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
