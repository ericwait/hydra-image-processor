#include "ScopedProcessMutex.h"

#include <stdexcept>

#ifdef USE_WINDOWS_IPC_MUTEX
#include <windows.h>

#undef min
#undef max

HANDLE ScopedProcessMutex::mutexHandle = NULL;

ScopedProcessMutex::ScopedProcessMutex(const char* name)
{
	if ( !mutexHandle )
	{
		mutexHandle = CreateMutex(NULL, false, name);
		if ( !mutexHandle && GetLastError() == ERROR_ACCESS_DENIED )
			mutexHandle = OpenMutex(SYNCHRONIZE, false, name);

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

#else
using boost::interprocess::named_mutex;
using boost::interprocess::open_or_create;

ScopedProcessMutex::ScopedProcessMutex(const char* name)
	: ipc_mutex(open_or_create, name)
{
	ipc_mutex.lock();
}

ScopedProcessMutex::~ScopedProcessMutex()
{
	ipc_mutex.unlock();
}

#endif
