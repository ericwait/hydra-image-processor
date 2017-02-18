#include "ScopedProcessMutex.h"

HANDLE ScopedProcessMutex::mutexHandle = NULL;

ScopedProcessMutex::ScopedProcessMutex(const std::string& name)
{
	if ( !mutexHandle )
	{
		mutexHandle = CreateMutex(NULL, false, name.c_str());
		if ( !mutexHandle && GetLastError() == ERROR_ACCESS_DENIED )
			mutexHandle = OpenMutex(SYNCHRONIZE, false, name.c_str());

		if ( !mutexHandle )
			throw std::runtime_error("Error creating mutex handle!");
	}

	DWORD waitResult = WaitForSingleObject(mutexHandle, INFINITE);
	if ( waitResult == WAIT_FAILED )
	{
		mutexHandle = NULL;
		throw std::runtime_error("Error unable to acquire mutex!");
	}
	else if ( waitResult == WAIT_ABANDONED )
	{
		mutexHandle = NULL;
		throw std::runtime_error("Previous thread terminated without releasing mutex!");
	}
}

ScopedProcessMutex::~ScopedProcessMutex()
{
	if ( mutexHandle )
		ReleaseMutex(mutexHandle);
}
