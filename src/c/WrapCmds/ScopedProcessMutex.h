#pragma once

#define USE_PROCESS_MUTEX
#ifdef USE_PROCESS_MUTEX
 #define SCOPED_PROCESS_MUTEX(Name) ScopedProcessMutex Name##_mutex(#Name)
#else
 #define SCOPED_PROCESS_MUTEX(Name)
#endif

#ifdef _WIN32
 #define USE_WINDOWS_IPC_MUTEX (1)
#endif

#ifndef USE_WINDOWS_IPC_MUTEX
	#define BOOST_DATE_TIME_NO_LIB (1)
	#include "boost/interprocess/sync/named_mutex.hpp"
#endif

class ScopedProcessMutex
{
public:
	ScopedProcessMutex(const char* name);

	// Cannot default-construct
	ScopedProcessMutex() = delete;

	// No move or copy semantics
	ScopedProcessMutex(const ScopedProcessMutex& other) = delete;
	ScopedProcessMutex& operator=(const ScopedProcessMutex& other) = delete;
	ScopedProcessMutex(ScopedProcessMutex&& other) = delete;
	ScopedProcessMutex& operator=(ScopedProcessMutex&& other) = delete;

	~ScopedProcessMutex();

private:

#ifdef USE_WINDOWS_IPC_MUTEX
	static void* mutexHandle;
#else
	boost::interprocess::named_mutex ipc_mutex;
#endif
};
