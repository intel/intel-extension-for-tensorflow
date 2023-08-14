/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef ITEX_CORE_OPS_UTILS_LOGGING_H_
#define ITEX_CORE_OPS_UTILS_LOGGING_H_

#include <atomic>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "itex/core/ops/utils/integral_types.h"
#include "itex/core/ops/utils/macros.h"

#undef ERROR

namespace itex {
const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace internal {

using std::string;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage() override;

  // Change the location of the log message.
  LogMessage& AtLocation(const char* fname, int line);

  // Returns the minimum log level for ITEX_VLOG statements.
  static int64 MinVLogLevel();

  // Returns whether ITEX_VLOG level lvl is activated for the file fname.
  static bool VmoduleActivated(const char* fname, int level);

 protected:
  void GenerateLogMessage();

  void IssueLink();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

// Uses the lower operator & precedence to voidify a LogMessage reference, so
// that the ternary ITEX_VLOG() implementation is balanced, type wise.
struct Voidifier {
  template <typename T>
  void operator&(const T&) const {}
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line) TF_ATTRIBUTE_COLD;
  TF_ATTRIBUTE_NORETURN ~LogMessageFatal() override;
};

// LogMessageNull supports the ITEX_DVLOG macro by simply dropping any log
// messages.
class LogMessageNull : public std::basic_ostringstream<char> {
 public:
  LogMessageNull() {}
  ~LogMessageNull() override {}
};

#define _ITEX_LOG_INFO \
  ::itex::internal::LogMessage(__FILE__, __LINE__, ::itex::INFO)
#define _ITEX_LOG_WARNING \
  ::itex::internal::LogMessage(__FILE__, __LINE__, ::itex::WARNING)
#define _ITEX_LOG_ERROR \
  ::itex::internal::LogMessage(__FILE__, __LINE__, ::itex::ERROR)
#define _ITEX_LOG_FATAL ::itex::internal::LogMessageFatal(__FILE__, __LINE__)

#define _ITEX_LOG_QFATAL _ITEX_LOG_FATAL

#define ITEX_LOG(severity) _ITEX_LOG_##severity

#ifdef IS_MOBILE_PLATFORM

// Turn ITEX_VLOG off when under mobile devices for considerations of binary
// size.
#define ITEX_VLOG_IS_ON(lvl) ((lvl) <= 0)

#else

// Otherwise, set TF_CPP_MIN_VLOG_LEVEL environment to update minimum log level
// of ITEX_VLOG, or TF_CPP_VMODULE to set the minimum log level for individual
// translation units.
#define ITEX_VLOG_IS_ON(lvl)                                          \
  (([](int level, const char* fname) {                                \
    static const bool vmodule_activated =                             \
        ::itex::internal::LogMessage::VmoduleActivated(fname, level); \
    return vmodule_activated;                                         \
  })(lvl, __FILE__))

#endif

#define ITEX_VLOG(level)                     \
  ITEX_PREDICT_TRUE(!ITEX_VLOG_IS_ON(level)) \
  ? (void)0                                  \
  : ::itex::internal::Voidifier() &          \
          ::itex::internal::LogMessage(__FILE__, __LINE__, itex::INFO)

// `ITEX_DVLOG` behaves like `ITEX_VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
// Otherwise, it compiles away and does nothing.
#ifndef NDEBUG
#define ITEX_DVLOG ITEX_VLOG
#else
#define ITEX_DVLOG(verbose_level) \
  while (false && (verbose_level) > 0) ::itex::internal::LogMessageNull()
#endif

class LogEveryNState {
 public:
  bool ShouldLog(int n);
  uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogFirstNState {
 public:
  bool ShouldLog(int n);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryPow2State {
 public:
  bool ShouldLog(int ignored);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
};

class LogEveryNSecState {
 public:
  bool ShouldLog(double seconds);
  uint32 counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32> counter_{0};
  // Cycle count according to CycleClock that we should next log at.
  std::atomic<int64> next_log_time_cycles_{0};
};

// This macro has a lot going on!
//
// * A local static (`logging_internal_stateful_condition_state`) is
//   declared in a scope such that each `ITEX_LOG_EVERY_N` (etc.) line has its
//   own state.
// * `COUNTER`, the third variable, is used to support `<< COUNTER`. It is not
//   mangled, so shadowing can be a problem, albeit more of a
//   shoot-yourself-in-the-foot one.  Don't name your variables `COUNTER`.
// * A single for loop can declare state and also test
//   `condition && state.ShouldLog()`, but there's no way to constrain it to run
//   only once (or not at all) without declaring another variable.  The outer
//   for-loop declares this variable (`do_log`).
// * Using for loops instead of if statements means there's no risk of an
//   ambiguous dangling else statement.
#define ITEX_LOGGING_INTERNAL_STATEFUL_CONDITION(kind, condition, arg) \
  for (bool logging_internal_stateful_condition_do_log(condition);     \
       logging_internal_stateful_condition_do_log;                     \
       logging_internal_stateful_condition_do_log = false)             \
    for (static ::itex::internal::Log##kind##State                     \
             logging_internal_stateful_condition_state;                \
         logging_internal_stateful_condition_do_log &&                 \
         logging_internal_stateful_condition_state.ShouldLog(arg);     \
         logging_internal_stateful_condition_do_log = false)           \
      for (const uint32_t COUNTER ABSL_ATTRIBUTE_UNUSED =              \
               logging_internal_stateful_condition_state.counter();    \
           logging_internal_stateful_condition_do_log;                 \
           logging_internal_stateful_condition_do_log = false)

// An instance of `ITEX_LOG_EVERY_N` increments a hidden zero-initialized
// counter every time execution passes through it and logs the specified message
// when the counter's value is a multiple of `n`, doing nothing otherwise.  Each
// instance has its own counter.  The counter's value can be logged by streaming
// the symbol `COUNTER`.  `ITEX_LOG_EVERY_N` is thread-safe.
// Example:
//
//   for (const auto& user : all_users) {
//     ITEX_LOG_EVERY_N(INFO, 1000) << "Processing user #" << COUNTER;
//     ProcessUser(user);
//   }
#define ITEX_LOG_EVERY_N(severity, n)                       \
  ITEX_LOGGING_INTERNAL_STATEFUL_CONDITION(EveryN, true, n) \
  ITEX_LOG(severity)
// `ITEX_LOG_FIRST_N` behaves like `ITEX_LOG_EVERY_N` except that the specified
// message is logged when the counter's value is less than `n`.
// `ITEX_LOG_FIRST_N` is thread-safe.
#define ITEX_LOG_FIRST_N(severity, n)                       \
  ITEX_LOGGING_INTERNAL_STATEFUL_CONDITION(FirstN, true, n) \
  ITEX_LOG(severity)
// `ITEX_LOG_EVERY_POW_2` behaves like `ITEX_LOG_EVERY_N` except that the
// specified message is logged when the counter's value is a power of 2.
// `ITEX_LOG_EVERY_POW_2` is thread-safe.
#define ITEX_LOG_EVERY_POW_2(severity)                         \
  ITEX_LOGGING_INTERNAL_STATEFUL_CONDITION(EveryPow2, true, 0) \
  ITEX_LOG(severity)
// An instance of `ITEX_LOG_EVERY_N_SEC` uses a hidden state variable to log the
// specified message at most once every `n_seconds`.  A hidden counter of
// executions (whether a message is logged or not) is also maintained and can be
// logged by streaming the symbol `COUNTER`.  `ITEX_LOG_EVERY_N_SEC` is
// thread-safe. Example:
//
//   ITEX_LOG_EVERY_N_SEC(INFO, 2.5) << "Got " << COUNTER << " cookies so far";
#define ITEX_LOG_EVERY_N_SEC(severity, n_seconds)                      \
  ITEX_LOGGING_INTERNAL_STATEFUL_CONDITION(EveryNSec, true, n_seconds) \
  ITEX_LOG(severity)

// ITEX_CHECK dies with a fatal error if condition is not true.  It is *not*
// controlled by NDEBUG, so the check will be executed regardless of
// compilation mode.  Therefore, it is safe to do things like:
//    ITEX_CHECK(fp->Write(x) == 4)
#define ITEX_CHECK(condition)           \
  if (ITEX_PREDICT_FALSE(!(condition))) \
  ITEX_LOG(FATAL) << "Check failed: " #condition " "

// Function is overloaded for integral types to allow static const
// integrals declared in classes and not defined to be used as arguments to
// ITEX_CHECK* macros. It's not encouraged though.
template <typename T>
inline const T& GetReferenceableValue(const T& t) {
  return t;
}
inline char GetReferenceableValue(char t) { return t; }
inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
inline signed char GetReferenceableValue(signed char t) { return t; }
inline int16 GetReferenceableValue(int16 t) { return t; }
inline uint16 GetReferenceableValue(uint16 t) { return t; }
inline int GetReferenceableValue(int t) { return t; }
inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
inline int64 GetReferenceableValue(int64 t) { return t; }
inline uint64 GetReferenceableValue(uint64 t) { return t; }

// This formats a value for a failing CHECK_XX statement.  Ordinarily,
// it uses the definition for operator<<, with a few special cases below.
template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

// Overrides for char types provide readable values for unprintable
// characters.
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v);
template <>
void MakeCheckOpValueString(
    std::ostream* os, const signed char& v);  // NOLINT(runtime/references)
template <>
void MakeCheckOpValueString(
    std::ostream* os, const unsigned char& v);  // NOLINT(runtime/references)

#if LANG_CXX11
// We need an explicit specialization for std::nullptr_t.
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v);
#endif

// A container for a string pointer which can be evaluated to a bool -
// true iff the pointer is non-NULL.
struct CheckOpString {
  explicit CheckOpString(string* str) : str_(str) {}
  // No destructor: if str_ is non-NULL, we're about to ITEX_LOG(FATAL),
  // so there's no point in cleaning up str_.
  explicit operator bool() const { return ITEX_PREDICT_FALSE(str_ != nullptr); }
  string* str_;
};

// Build the error message string. Specify no inlining for code size.
template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2,
                          const char* exprtext) TF_ATTRIBUTE_NOINLINE;

// A helper class for formatting "expr (V1 vs. V2)" in a CHECK_XX
// statement.  See MakeCheckOpString for sample usage.  Other
// approaches were considered: use of a template method (e.g.,
// base::BuildCheckOpString(exprtext, base::Print<T1>, &v1,
// base::Print<T2>, &v2), however this approach has complications
// related to volatile arguments and function-pointer arguments).
class CheckOpMessageBuilder {
 public:
  // Inserts "exprtext" and " (" to the stream.
  explicit CheckOpMessageBuilder(const char* exprtext);
  // Deletes "stream_".
  ~CheckOpMessageBuilder();
  // For inserting the first variable.
  std::ostream* ForVar1() { return stream_; }
  // For inserting the second variable (adds an intermediate " vs. ").
  std::ostream* ForVar2();
  // Get the result (inserts the closing ")").
  string* NewString();

 private:
  std::ostringstream* stream_;
};

template <typename T1, typename T2>
string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

// Helper functions for ITEX_CHECK_OP macro.
// The (int, int) specialization works around the issue that the compiler
// will not instantiate the template version of the function on values of
// unnamed enum type - see comment below.
// The (size_t, int) and (int, size_t) specialization are to handle unsigned
// comparison errors while still being thorough with the comparison.
#define ITEX_DEFINE_CHECK_OP_IMPL(name, op)                          \
  template <typename T1, typename T2>                                \
  inline string* name##Impl(const T1& v1, const T2& v2,              \
                            const char* exprtext) {                  \
    if (ITEX_PREDICT_TRUE(v1 op v2))                                 \
      return nullptr;                                                \
    else                                                             \
      return ::itex::internal::MakeCheckOpString(v1, v2, exprtext);  \
  }                                                                  \
  inline string* name##Impl(int v1, int v2, const char* exprtext) {  \
    return name##Impl<int, int>(v1, v2, exprtext);                   \
  }                                                                  \
  inline string* name##Impl(const size_t v1, const int v2,           \
                            const char* exprtext) {                  \
    if (ITEX_PREDICT_FALSE(v2 < 0)) {                                \
      return ::itex::internal::MakeCheckOpString(v1, v2, exprtext);  \
    }                                                                \
    return name##Impl<size_t, size_t>(v1, v2, exprtext);             \
  }                                                                  \
  inline string* name##Impl(const int v1, const size_t v2,           \
                            const char* exprtext) {                  \
    if (ITEX_PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) { \
      return ::itex::internal::MakeCheckOpString(v1, v2, exprtext);  \
    }                                                                \
    const size_t uval = static_cast<size_t>((unsigned)v2);           \
    return name##Impl<size_t, size_t>(v1, uval, exprtext);           \
  }

// We use the full name ITEX_Check_EQ, ITEX_Check_NE, etc. in case the file
// including base/logging.h provides its own #defines for the simpler names EQ,
// NE, etc. This happens if, for example, those are used as token names in a
// yacc grammar.
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_EQ,
                          ==)  // Compilation error with ITEX_CHECK_EQ(NULL, x)?
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_NE,
                          !=)  // Use ITEX_CHECK(x == NULL) instead.
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_LE, <=)
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_LT, <)
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_GE, >=)
ITEX_DEFINE_CHECK_OP_IMPL(ITEX_Check_GT, >)
#undef ITEX_DEFINE_CHECK_OP_IMPL

// In optimized mode, use CheckOpString to hint to compiler that
// the while condition is unlikely.
#define ITEX_CHECK_OP_LOG(name, op, val1, val2)                                \
  while (::itex::internal::CheckOpString _result{::itex::internal::name##Impl( \
      ::itex::internal::GetReferenceableValue(val1),                           \
      ::itex::internal::GetReferenceableValue(val2),                           \
      #val1 " " #op " " #val2)})                                               \
  ::itex::internal::LogMessageFatal(__FILE__, __LINE__) << *(_result.str_)

#define ITEX_CHECK_OP(name, op, val1, val2) \
  ITEX_CHECK_OP_LOG(name, op, val1, val2)

// ITEX_CHECK_EQ/NE/...
#define ITEX_CHECK_EQ(val1, val2) ITEX_CHECK_OP(ITEX_Check_EQ, ==, val1, val2)
#define ITEX_CHECK_NE(val1, val2) ITEX_CHECK_OP(ITEX_Check_NE, !=, val1, val2)
#define ITEX_CHECK_LE(val1, val2) ITEX_CHECK_OP(ITEX_Check_LE, <=, val1, val2)
#define ITEX_CHECK_LT(val1, val2) ITEX_CHECK_OP(ITEX_Check_LT, <, val1, val2)
#define ITEX_CHECK_GE(val1, val2) ITEX_CHECK_OP(ITEX_Check_GE, >=, val1, val2)
#define ITEX_CHECK_GT(val1, val2) ITEX_CHECK_OP(ITEX_Check_GT, >, val1, val2)
#define ITEX_CHECK_NOTNULL(val)                      \
  ::itex::internal::CheckNotNull(__FILE__, __LINE__, \
                                 "'" #val "' Must be non NULL", (val))

#ifndef NDEBUG
// ITEX_DCHECK_EQ/NE/...
#define ITEX_DCHECK(condition) ITEX_CHECK(condition)
#define ITEX_DCHECK_EQ(val1, val2) ITEX_CHECK_EQ(val1, val2)
#define ITEX_DCHECK_NE(val1, val2) ITEX_CHECK_NE(val1, val2)
#define ITEX_DCHECK_LE(val1, val2) ITEX_CHECK_LE(val1, val2)
#define ITEX_DCHECK_LT(val1, val2) ITEX_CHECK_LT(val1, val2)
#define ITEX_DCHECK_GE(val1, val2) ITEX_CHECK_GE(val1, val2)
#define ITEX_DCHECK_GT(val1, val2) ITEX_CHECK_GT(val1, val2)

#else

#define ITEX_DCHECK(condition) \
  while (false && (condition)) ITEX_LOG(FATAL)

// NDEBUG is defined, so ITEX_DCHECK_EQ(x, y) and so on do nothing.
// However, we still want the compiler to parse x and y, because
// we don't want to lose potentially useful errors and warnings.
// _DCHECK_NOP is a helper, and should not be used outside of this file.
#define _ITEX_DCHECK_NOP(x, y) \
  while (false && ((void)(x), (void)(y), 0)) ITEX_LOG(FATAL)

#define ITEX_DCHECK_EQ(x, y) _ITEX_DCHECK_NOP(x, y)
#define ITEX_DCHECK_NE(x, y) _ITEX_DCHECK_NOP(x, y)
#define ITEX_DCHECK_LE(x, y) _ITEX_DCHECK_NOP(x, y)
#define ITEX_DCHECK_LT(x, y) _ITEX_DCHECK_NOP(x, y)
#define ITEX_DCHECK_GE(x, y) _ITEX_DCHECK_NOP(x, y)
#define ITEX_DCHECK_GT(x, y) _ITEX_DCHECK_NOP(x, y)

#endif

// These are for when you don't want a ITEX_CHECK failure to print a verbose
// stack trace.  The implementation of ITEX_CHECK* in this file already doesn't.
#define ITEX_QCHECK(condition) ITEX_CHECK(condition)
#define ITEX_QCHECK_EQ(x, y) ITEX_CHECK_EQ(x, y)
#define ITEX_QCHECK_NE(x, y) ITEX_CHECK_NE(x, y)
#define ITEX_QCHECK_LE(x, y) ITEX_CHECK_LE(x, y)
#define ITEX_QCHECK_LT(x, y) ITEX_CHECK_LT(x, y)
#define ITEX_QCHECK_GE(x, y) ITEX_CHECK_GE(x, y)
#define ITEX_QCHECK_GT(x, y) ITEX_CHECK_GT(x, y)

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << string(exprtext);
  }
  return std::forward<T>(t);
}

int64 MinLogLevelFromEnv();

int64 MinVLogLevelFromEnv();

}  // namespace internal

// LogSink support adapted from //base/logging.h
//
// `LogSink` is an interface which can be extended to intercept and process
// all log messages. LogSink implementations must be thread-safe. A single
// instance will be called from whichever thread is performing a logging
// operation.
class TFLogEntry {
  static absl::LogSeverity AsAbslLogSeverity(int severity) {
    return static_cast<absl::LogSeverity>(severity);
  }

 public:
  explicit TFLogEntry(int severity, absl::string_view log_line)
      : severity_(AsAbslLogSeverity(severity)), log_line_(log_line) {}

  absl::LogSeverity log_severity() const { return severity_; }
  std::string ToString() const { return std::string(log_line_); }

 private:
  const absl::LogSeverity severity_;
  const absl::string_view log_line_;
};

}  // namespace itex

#endif  // ITEX_CORE_OPS_UTILS_LOGGING_H_