#include "CWrappers.h"
#include "ImageView.h"
#include "Vec.h"

#include <cstdio>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>

// Simple minimal unit testing framework
#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("[FAILED] %s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond); \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("[FAILED] %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return false; \
        } \
    } while(0)

#define RUN_TEST(func) \
    do { \
        printf("Running %s... ", #func); \
        if (func()) { \
            printf("[OK]\n"); \
        } else { \
            allPassed = false; \
        } \
    } while(0)

bool test_gaussian()
{
    // Keep the impulse's Gaussian support (~3*sigma) away from every face:
    // the filter renormalizes truncated kernels at the borders, so energy is
    // only preserved while the smoothed impulse stays interior.
    Vec<std::size_t> dims = {64, 64, 32};
    ImageOwner<float> imIn(0, dims, 1, 1);
    ImageOwner<float> imOut(0, dims, 1, 1);

    // Fill with a impulse in the middle
    std::fill(imIn.getPtr(), imIn.getPtr() + imIn.getNumElements(), 0.0f);
    ImageDimensions midCoord(Vec<std::size_t>(32, 32, 16), 0, 0);
    imIn.getPtr()[imIn.getDims().linearAddressAt(midCoord)] = 100.0f;

    Vec<double> sigmas(2.0, 2.0, 2.0);
    ImageView<float> imOutView(imOut);
    CudaCall_Gaussian::run(imIn, imOutView, sigmas, 1, 0);

    // Check if it smoothed (sum should be preserved approximately)
    double sumIn = 0;
    double sumOut = 0;
    for(size_t i=0; i<imIn.getNumElements(); ++i) sumIn += imIn.getPtr()[i];
    for(size_t i=0; i<imOut.getNumElements(); ++i) sumOut += imOut.getPtr()[i];

    TEST_ASSERT_MSG(std::abs(sumIn - sumOut) <= 1.0f, "Gaussian filter should preserve total energy (approx)");

    // The impulse must actually have been spread out
    float peakOut = imOut.getPtr()[imOut.getDims().linearAddressAt(midCoord)];
    TEST_ASSERT_MSG(peakOut > 0.0f && peakOut < 100.0f, "Gaussian filter should smooth the impulse peak");

    return true;
}

bool test_min_max()
{
    Vec<std::size_t> dims = {32, 32, 10};
    ImageOwner<uint8_t> imIn(0, dims, 1, 1);
    
    // Pattern that covers range
    for(size_t i=0; i<imIn.getNumElements(); ++i) imIn.getPtr()[i] = (uint8_t)(i % 256);
    // Explicitly set some values to ensure we hit boundaries (though logic above guarantees it for large enough img)
    imIn.getPtr()[10] = 5;
    imIn.getPtr()[20] = 250;
    
    uint8_t minVal, maxVal;
    CudaCall_GetMinMax::run(imIn, minVal, maxVal, 0);
    
    TEST_ASSERT_MSG(minVal == 0, "Minimum value should be 0");
    TEST_ASSERT_MSG(maxVal == 255, "Maximum value should be 255");
    
    return true;
}

int main()

{

    // Check for CUDA devices

    int devCount = deviceCount();

    if (devCount <= 0) {

        printf("No CUDA devices found (count=%d). Skipping accuracy tests.\n", devCount);

        return 0;

    }



    bool allPassed = true;

    

    printf("Starting C++ Accuracy Tests on %d device(s)\n", devCount);

    printf("===========================\n");

    

    RUN_TEST(test_gaussian);

    RUN_TEST(test_min_max);

    

    printf("\nTest Summary: %s\n", allPassed ? "PASSED" : "FAILED");

    

    return allPassed ? 0 : 1;

}
