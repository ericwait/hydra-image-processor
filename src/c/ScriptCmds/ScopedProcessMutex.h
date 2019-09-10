#pragma once

#ifdef USE_PROCESS_MUTEX
 #define SCOPED_PROCESS_MUTEX(Name) ScopedProcessMutex Name##_mutex(#Name)
#else
 #define SCOPED_PROCESS_MUTEX(Name)
#endif

#ifdef _WIN32
 #define USE_WINDOWS_IPC_MUTEX (1)
#elif defined(__linux__)
 #define USE_PTHREADS_ROBUST_MUTEX (1)
#else
 #define USE_BOOST_IPC_MUTEX (1)
#endif

#if defined(USE_PTHREADS_ROBUST_MUTEX)
 #include <memory>
#elif defined(USE_BOOST_IPC_MUTEX)
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

	// Allow force release of mutex resource (cross-process removal)
	static void remove(const char* name);

private:

#if defined(USE_WINDOWS_IPC_MUTEX)
	static void* mutexHandle;
#elif defined(USE_PTHREADS_ROBUST_MUTEX)
	struct PThreadMutex;
	static thread_local std::unique_ptr<PThreadMutex> procMutex;
#else
	boost::interprocess::named_mutex ipc_mutex;
#endif
};
