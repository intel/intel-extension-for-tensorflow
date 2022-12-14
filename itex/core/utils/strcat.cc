/* Copyright (c) 2021 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "itex/core/utils/strcat.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "absl/meta/type_traits.h"
#include "itex/core/utils/logging.h"

namespace itex {
namespace strings {

AlphaNum::AlphaNum(Hex hex) {
  char* const end = &digits_[kFastToBufferSize];
  char* writer = end;
  uint64 value = hex.value;
  uint64 width = hex.spec;
  // We accomplish minimum width by OR'ing in 0x10000 to the user's value,
  // where 0x10000 is the smallest hex number that is as wide as the user
  // asked for.
  uint64 mask = (static_cast<uint64>(1) << (width - 1) * 4) | value;
  static const char hexdigits[] = "0123456789abcdef";
  do {
    *--writer = hexdigits[value & 0xF];
    value >>= 4;
    mask >>= 4;
  } while (mask != 0);
  piece_ = StringPiece(writer, end - writer);
}

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or integers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, and integer values.
// ----------------------------------------------------------------------

// Append is merely a version of std::copy that returns the address of the byte
// after the area just overwritten.  It comes in multiple flavors to minimize
// call overhead.
static char* Append1(char* out, const AlphaNum& x) {
  std::copy(x.data(), x.data() + x.size(), out);
  return out + x.size();
}

static char* Append2(char* out, const AlphaNum& x1, const AlphaNum& x2) {
  std::copy(x1.data(), x1.data() + x1.size(), out);
  out += x1.size();

  std::copy(x2.data(), x2.data() + x2.size(), out);
  return out + x2.size();
}

static char* Append4(char* out, const AlphaNum& x1, const AlphaNum& x2,
                     const AlphaNum& x3, const AlphaNum& x4) {
  std::copy(x1.data(), x1.data() + x1.size(), out);
  out += x1.size();

  std::copy(x2.data(), x2.data() + x2.size(), out);
  out += x2.size();

  std::copy(x3.data(), x3.data() + x3.size(), out);
  out += x3.size();

  std::copy(x4.data(), x4.data() + x4.size(), out);
  return out + x4.size();
}

string StrCat(const AlphaNum& a) { return string(a.data(), a.size()); }

string StrCat(const AlphaNum& a, const AlphaNum& b) {
  string result(a.size() + b.size(), '\0');
  char* const begin = &*result.begin();
  char* out = Append2(begin, a, b);
  ITEX_DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum& a, const AlphaNum& b, const AlphaNum& c) {
  string result(a.size() + b.size() + c.size(), '\0');
  char* const begin = &*result.begin();
  char* out = Append2(begin, a, b);
  out = Append1(out, c);
  ITEX_DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum& a, const AlphaNum& b, const AlphaNum& c,
              const AlphaNum& d) {
  string result(a.size() + b.size() + c.size() + d.size(), '\0');
  char* const begin = &*result.begin();
  char* out = Append4(begin, a, b, c, d);
  ITEX_DCHECK_EQ(out, begin + result.size());
  return result;
}

namespace {
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) { s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};

static inline void STLStringResizeUninitialized(string* s, size_t new_size) {
  ResizeUninitializedTraits<string>::Resize(s, new_size);
}

}  // namespace
namespace internal {

// Do not call directly - these are not part of the public API.
string CatPieces(std::initializer_list<StringPiece> pieces) {
  size_t total_size = 0;
  for (const StringPiece piece : pieces) total_size += piece.size();
  string result(total_size, '\0');

  char* const begin = &*result.begin();
  char* out = begin;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    std::copy(piece.data(), piece.data() + this_size, out);
    out += this_size;
  }
  ITEX_DCHECK_EQ(out, begin + result.size());
  return result;
}

// It's possible to call StrAppend with a StringPiece that is itself a fragment
// of the string we're appending to.  However the results of this are random.
// Therefore, check for this in debug mode.  Use unsigned math so we only have
// to do one comparison.
#define DCHECK_NO_OVERLAP(dest, src)                      \
  ITEX_DCHECK_GE(uintptr_t((src).data() - (dest).data()), \
                 uintptr_t((dest).size()))

void AppendPieces(string* result, std::initializer_list<StringPiece> pieces) {
  size_t old_size = result->size();
  size_t total_size = old_size;
  for (const StringPiece piece : pieces) {
    DCHECK_NO_OVERLAP(*result, piece);
    total_size += piece.size();
  }
  STLStringResizeUninitialized(result, total_size);

  char* const begin = &*result->begin();
  char* out = begin + old_size;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    std::copy(piece.data(), piece.data() + this_size, out);
    out += this_size;
  }
  ITEX_DCHECK_EQ(out, begin + result->size());
}

}  // namespace internal

void StrAppend(string* result, const AlphaNum& a) {
  DCHECK_NO_OVERLAP(*result, a);
  result->append(a.data(), a.size());
}

void StrAppend(string* result, const AlphaNum& a, const AlphaNum& b) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(result, old_size + a.size() + b.size());
  char* const begin = &*result->begin();
  char* out = Append2(begin + old_size, a, b);
  ITEX_DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string* result, const AlphaNum& a, const AlphaNum& b,
               const AlphaNum& c) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(result,
                               old_size + a.size() + b.size() + c.size());
  char* const begin = &*result->begin();
  char* out = Append2(begin + old_size, a, b);
  out = Append1(out, c);
  ITEX_DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string* result, const AlphaNum& a, const AlphaNum& b,
               const AlphaNum& c, const AlphaNum& d) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  DCHECK_NO_OVERLAP(*result, d);
  string::size_type old_size = result->size();
  STLStringResizeUninitialized(
      result, old_size + a.size() + b.size() + c.size() + d.size());
  char* const begin = &*result->begin();
  char* out = Append4(begin + old_size, a, b, c, d);
  ITEX_DCHECK_EQ(out, begin + result->size());
}

}  // namespace strings
}  // namespace itex
