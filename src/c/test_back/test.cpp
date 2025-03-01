#include "CWrappers.h"
#include "ImageView.h"

#include <cstdio>
#include <random>

int main(void)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rnddist(0,255);

    ImageOwner<uint8_t> imIn(20, {200,200,20}, 1,1);
    for (int i=0; i < 200*200*20; ++i)
    {
        imIn.getPtr()[i] = rnddist(rng);
    }
    // ImageOwner<uint8_t> imOut(0, {200,200,20}, 1,1);

    uint8_t outMin;
    uint8_t outMax;
    CudaCall_GetMinMax::run(imIn, outMin, outMax, 0);
    // CudaCall_GetMinMax_Stub::cGetMinMax_stub(imIn, outMin, outMax, 0);

    printf("Min/max: %d, %d\n", outMin, outMax);

    return 0;
}
