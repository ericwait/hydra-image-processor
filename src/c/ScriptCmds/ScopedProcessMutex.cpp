#include "ScopedProcessMutex.h"

#include <stdexcept>
#include <memory>

// Helpers for getting user ID
std::string getProcessUser();

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <sddl.h>

#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX

struct LocalFreeFunc { inline void operator() (HLOCAL* ptr) { LocalFree((HLOCAL)ptr); } };
template <typename T> using LocalUnique = std::unique_ptr<T, LocalFreeFunc>;

std::string getProcessUser()
{
	HANDLE hToken;

	HANDLE hProc = GetCurrentProcess();
	if ( !OpenProcessToken(hProc, TOKEN_QUERY, &hToken) )
		return "unk";

	DWORD dwSize = sizeof(TOKEN_USER);
	TOKEN_USER tu = { 0 };

	if ( !GetTokenInformation(hToken, TokenUser, &tu, dwSize, &dwSize) )
		return "unk";

	char* strSID = nullptr;
	if ( !ConvertSidToStringSidA(&tu.User.Sid, &strSID) )
		return "unk";

	std::string outStr(strSID);
	LocalFree(strSID);

	return outStr;
}

#elif defined(__linux__)
#include <unistd.h>
#include <sys/types.h>

std::string getProcessUser()
{
	uid_t uid = geteuid();

	const int MAXLEN = 10;
	char uidStr[MAXLEN+1];
	snprintf(uidStr, MAXLEN, "%d", uid);

	return uidStr;
}

#endif


#if defined(USE_WINDOWS_IPC_MUTEX)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX


HANDLE ScopedProcessMutex::mutexHandle = nullptr;

ScopedProcessMutex::ScopedProcessMutex(const char* name)
{
	if ( !mutexHandle )
	{
		// Postfix a unique user-id to the mutex name
		std::string mtx_name = name + getProcessUser();

		mutexHandle = CreateMutex(NULL, false, mtx_name.c_str());
		if ( !mutexHandle && GetLastError() == ERROR_ACCESS_DENIED )
			mutexHandle = OpenMutex(SYNCHRONIZE, false, mtx_name.c_str());

		if ( !mutexHandle )
			throw std::runtime_error("Error creating mutex handle!");
	}

	DWORD waitResult = WaitForSingleObject(mutexHandle, INFINITE);
	if ( waitResult == WAIT_FAILED )
	{
		mutexHandle = NULL;
		throw std::runtime_error("Error unable to acquire mutex!");
	}
	// MW - Treat a previous crash as ok since the GPU is likely to recover 
	//      from process crashes at the driver level (no longer throw error)
	else if ( waitResult == WAIT_ABANDONED )
	{}
}

ScopedProcessMutex::~ScopedProcessMutex()
{
	if ( mutexHandle )
		ReleaseMutex(mutexHandle);
}

#elif defined(USE_PTHREADS_ROBUST_MUTEX)
#include <errno.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <unistd.h>
#include <stdlib.h>

#include <thread>
#include <atomic>

#if (ATOMIC_INT_LOCK_FREE != 2)
 #error USE_PTHREADS_ROBUST_MUTEX implementation requires always lock-free atomic int type
#endif

struct ScopedProcessMutex::PThreadMutex
{
	enum SharedMutexState
	{
		Uninitialized = 0,
		Initializing = 1,
		Valid = 2,
	};

	struct SharedMemMutex
	{
		std::atomic_int state;
		pthread_mutex_t pthread_mutex;
	};

	int shm_fd;
	SharedMemMutex* sharedMem;
	std::string shm_name;

	PThreadMutex(const char* name)
		: shm_fd(-1), sharedMem(nullptr), shm_name(name)
	{
		try_create_mutex();
	}

	~PThreadMutex()
	{
		cleanup();
	}

	static void force_unlink(const char* name)
	{
		shm_unlink(name);
	}

	void lock()
	{
		int err = pthread_mutex_lock(&sharedMem->pthread_mutex);
		if ( err == EOWNERDEAD )
			err = pthread_mutex_consistent(&sharedMem->pthread_mutex);
		else if ( err != 0 )
			throw std::runtime_error("Error unable to acquire mutex!");
	}

	void unlock()
	{
		int err = pthread_mutex_unlock(&sharedMem->pthread_mutex);
		if (err != 0)
			throw std::runtime_error("Error unable to release mutex!");
	}

private:
	void try_create_mutex()
	{
		errno = 0;

		// Try to create shared memory-mapping
		shm_fd = shm_open(shm_name.c_str(), O_RDWR|O_CREAT|O_EXCL, S_IRUSR|S_IWUSR);
		if (shm_fd < 0)
		{
			if ( errno == EEXIST )
			{
				try_open_mutex();
				return;
			}
			else
				throw std::runtime_error("Error unable to create shared memory");
		}

		int err = ftruncate(shm_fd, sizeof(SharedMemMutex));
		if (err)
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Error unable to resize shared memory for mutex");
		}

		void* mapPtr = mmap(nullptr, sizeof(SharedMemMutex), PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
		if (mapPtr == MAP_FAILED)
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Failed to map shared memory for mutex");
		}

		sharedMem = (SharedMemMutex*) mapPtr;
		sharedMem->state.store(SharedMutexState::Initializing, std::memory_order_seq_cst);
		////// Guard other processes from using mutex until it's initialized
		std::unique_ptr<pthread_mutexattr_t, int(*)(pthread_mutexattr_t*)> mtxAttr(new pthread_mutexattr_t(),pthread_mutexattr_destroy);

		err = pthread_mutexattr_init(mtxAttr.get());
		if ( err )
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Error failed to initialize mutex attribute");
		}

		err = pthread_mutexattr_setpshared(mtxAttr.get(), PTHREAD_PROCESS_SHARED);
		if ( err )
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Error failed to set mutex shared");
		}

		err = pthread_mutexattr_setrobust(mtxAttr.get(), PTHREAD_MUTEX_ROBUST);
		if ( err )
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Error failed to set mutex robust");
		}

		err = pthread_mutex_init(&sharedMem->pthread_mutex, mtxAttr.get());
		if ( err )
		{
			err_create_cleanup(shm_name.c_str());
			throw std::runtime_error("Error failed to initialize mutex");
		}
		//////
		sharedMem->state.store(SharedMutexState::Valid, std::memory_order_seq_cst);
	}

	void try_open_mutex()
	{
		shm_fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR|S_IWUSR);
		if (shm_fd < 0)
			throw std::runtime_error("Error unable to open shared memory");

		struct stat shm_stat;

		const int chkLimit = 100;
		for (int i = 0; i < chkLimit; ++i)
		{
			// Wait for the shared-mem to be properly resized (ftruncate)
			int err = fstat(shm_fd, &shm_stat);
			if (err)
			{
				cleanup();
				throw std::runtime_error("Error unable to stat shared memory");
			}

			if (shm_stat.st_size > 0)
				break;

			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		// Timeout failure
		if (shm_stat.st_size == 0)
		{
			cleanup();
			throw std::runtime_error("Error timeout waiting for shared memory init");
		}

		void* mapPtr = mmap(nullptr, sizeof(SharedMemMutex), PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
		if (mapPtr == MAP_FAILED)
		{
			cleanup();
			throw std::runtime_error("Failed to map shared memory for mutex");
		}

		sharedMem = (SharedMemMutex*)mapPtr;

		// NOTE: sharedMem is already valid but mutex may not have been properly initialized yet
		int chkState;
		for (int i = 0; i < chkLimit; ++i)
		{
			// Wait for the mutex to be properly initialized
			chkState = sharedMem->state.load();
			if (chkState == SharedMutexState::Valid)
				break;

			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		// Timout failure
		if (chkState != SharedMutexState::Valid)
		{
			cleanup();
			throw std::runtime_error("Error timeout waiting for mutex init");
		}
	}

	// Cleanup helpers
	inline void err_create_cleanup(const char* name)
	{
		// Cleanup if creating-process errors
		safe_destroy_mutex();
		safe_unmap_mem();
		safe_close_shm();
		force_unlink(name);
	}

	// Normal cleanup remove local resources on create error
	inline void cleanup()
	{
		safe_unmap_mem();
		safe_close_shm();
	}

	inline void safe_destroy_mutex()
	{
		if (sharedMem)
		{
			int chkValid = SharedMutexState::Valid;
			if (sharedMem->state.compare_exchange_weak(chkValid, SharedMutexState::Uninitialized))
			{
				pthread_mutex_destroy(&sharedMem->pthread_mutex);
			}
		}
	}

	inline void safe_unmap_mem()
	{
		if (sharedMem)
		{
			munmap((void*)sharedMem, sizeof(SharedMemMutex));
			sharedMem = nullptr;
		}
	}

	inline void safe_close_shm()
	{
		// NOTE: Unlike safe_unlink_shm, this just closes the file descriptor
		//       it will not invalidate the shared memory for other processes
		if (shm_fd >= 0)
		{
			close(shm_fd);
			shm_fd = -1;
		}
	}
};

thread_local std::unique_ptr<ScopedProcessMutex::PThreadMutex> ScopedProcessMutex::procMutex = nullptr;

ScopedProcessMutex::ScopedProcessMutex(const char* name)
{
	if ( !procMutex)
	{
		// Postfix a unique user-id to the mutex name
		std::string mtx_name = std::string("/") + name + getProcessUser();
		procMutex = std::unique_ptr<PThreadMutex>(new PThreadMutex(mtx_name.c_str()));
	}

	if ( !procMutex )
		throw std::runtime_error("Error unable to open/create shared mutex!");

	procMutex->lock();
}

ScopedProcessMutex::~ScopedProcessMutex()
{
	if (procMutex)
		procMutex->unlock();
}

void ScopedProcessMutex::remove(const char* name)
{
	std::string mtxName = name + getProcessUser();
	PThreadMutex::force_unlink(mtxName.c_str());
}


#elif defined(USE_BOOST_IPC_MUTEX)
using boost::interprocess::named_mutex;
using boost::interprocess::open_or_create;

ScopedProcessMutex::ScopedProcessMutex(const char* name)
	: ipc_mutex(open_or_create, std::string(name + getProcessUser()).c_str())
{
	ipc_mutex.lock();
}

ScopedProcessMutex::~ScopedProcessMutex()
{
	ipc_mutex.unlock();
}

void ScopedProcessMutex::remove(const char* name)
{
	std::string mtxName = name + getProcessUser();
	named_mutex::remove(mtxName.c_str());
}

#endif
