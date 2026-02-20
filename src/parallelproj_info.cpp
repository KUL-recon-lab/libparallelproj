#include "parallelproj.h"

int parallelproj_cuda_enabled(void)
{
#if PARALLELPROJ_CUDA
  return 1;
#else
  return 0;
#endif
}

const char *parallelproj_version(void)
{
#ifdef PROJECT_VERSION
  return PROJECT_VERSION;
#else
  return "unknown";
#endif
}

int parallelproj_version_major(void)
{
#ifdef PROJECT_VERSION_MAJOR
  return PROJECT_VERSION_MAJOR;
#else
  return 0;
#endif
}

int parallelproj_version_minor(void)
{
#ifdef PROJECT_VERSION_MINOR
  return PROJECT_VERSION_MINOR;
#else
  return 0;
#endif
}

int parallelproj_version_patch(void)
{
#ifdef PROJECT_VERSION_PATCH
  return PROJECT_VERSION_PATCH;
#else
  return 0;
#endif
}
