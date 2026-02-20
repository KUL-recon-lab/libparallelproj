#include "parallelproj.h"

int parallelproj_cuda_enabled(void)
{
#if PARALLELPROJ_CUDA
    return 1;
#else
    return 0;
#endif
}
