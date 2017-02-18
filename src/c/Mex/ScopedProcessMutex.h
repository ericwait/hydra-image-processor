#pragma once

#include <string>
#include <windows.h>

class ScopedProcessMutex
{
public:
	ScopedProcessMutex(const std::string& name);
	~ScopedProcessMutex();

private:
	ScopedProcessMutex(){}
	ScopedProcessMutex(const ScopedProcessMutex& other){}

	static HANDLE mutexHandle;
};
