#pragma once
#include <string>

struct DevStats
{
    std::string name;
    int major;
    int minor;

    std::size_t constMem;
    std::size_t sharedMem;
    std::size_t totalMem;

    bool tccDriver;
    int mpCount;
    int threadsPerMP;

    int warpSize;
    int maxThreads;
};

int cDeviceStats(DevStats** stats);
std::size_t memoryAvailable(int device, std::size_t* totalOut = NULL);
bool checkFreeMemory(std::size_t needed, int device, bool throws = false);

