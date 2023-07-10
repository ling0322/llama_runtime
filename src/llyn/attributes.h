#pragma once

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define LY_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define LY_HAS_CPP_ATTRIBUTE(x) 0
#endif

#if LY_HAS_CPP_ATTRIBUTE(clang::lifetimebound)
#define LY_LIFETIME_BOUND [[clang::lifetimebound]]
#else
#define LY_LIFETIME_BOUND
#endif
