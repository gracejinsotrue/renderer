// Platform shims for the test programs.
//
// The engine and the kernels are already portable -- the only thing the move
// off WSL turned up in src/ was M_PI, which CMake handles with
// _USE_MATH_DEFINES. This header exists for the tests, which reach for POSIX
// where MSVC has its own spelling.
#pragma once

#ifdef _WIN32

#include <stdlib.h>

// setenv is POSIX; the MSVC CRT spells it _putenv_s, with the value and name
// in the same order and 0 for success either way. The overwrite flag has no
// counterpart because _putenv_s always overwrites, which is what every call
// site here passes anyway.
static inline int setenv(const char *name, const char *value, int /*overwrite*/)
{
    return (int)_putenv_s(name, value);
}

#else

#include <stdlib.h>

#endif
