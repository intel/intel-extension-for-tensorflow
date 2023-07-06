/* Copyright (c) 2023 Intel Corporation

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ITEX_CORE_UTILS_RANDOM_INPUTSTREAM_H_
#define ITEX_CORE_UTILS_RANDOM_INPUTSTREAM_H_

#include "itex/core/utils/cord.h"
#include "itex/core/utils/file_system.h"
#include "itex/core/utils/inputstream_interface.h"

namespace itex {
namespace io {

// Wraps a RandomAccessFile in an InputStreamInterface. A given instance of
// RandomAccessInputStream is NOT safe for concurrent use by multiple threads.
class RandomAccessInputStream : public InputStreamInterface {
 public:
  // Does not take ownership of 'file' unless owns_file is set to true. 'file'
  // must outlive *this.
  explicit RandomAccessInputStream(RandomAccessFile* file,
                                   bool owns_file = false);

  ~RandomAccessInputStream();

  Status ReadNBytes(int64_t bytes_to_read, tstring* result) override;

#if defined(TF_CORD_SUPPORT)
  Status ReadNBytes(int64_t bytes_to_read, absl::Cord* result) override;
#endif

  Status SkipNBytes(int64_t bytes_to_skip) override;

  int64_t Tell() const override;

  Status Seek(int64_t position) {
    pos_ = position;
    return Status::OK();
  }

  Status Reset() override { return Seek(0); }

 private:
  RandomAccessFile* file_;  // Not owned.
  int64_t pos_ = 0;         // Tracks where we are in the file.
  bool owns_file_ = false;
};

}  // namespace io
}  // namespace itex

#endif  // ITEX_CORE_UTILS_RANDOM_INPUTSTREAM_H_
