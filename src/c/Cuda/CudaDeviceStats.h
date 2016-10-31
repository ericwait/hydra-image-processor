#pragma once
#include <string>

struct DevStats
{
    std::string name;
    int major;
    int minor;

    size_t constMem;
    size_t sharedMem;
    size_t totalMem;

    bool tccDriver;
    int mpCount;
    int threadsPerMP;

    int warpSize;
    int maxThreads;
};

int cDeviceStats(DevStats** stats);
