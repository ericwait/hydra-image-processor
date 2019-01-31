#pragma once

#ifdef USE_PROCESS_MUTEX
	#define BOOST_DATE_TIME_NO_LIB (1)
	#include "boost/interprocess/sync/scoped_lock.hpp"
	#include "boost/interprocess/sync/named_mutex.hpp"

	using boost::interprocess::scoped_lock;
	using boost::interprocess::named_mutex;
	using boost::interprocess::open_or_create;

	#define SCOPED_MUTEX(Name) scoped_lock<named_mutex> Name##_lock(named_mutex(open_or_create, #Name))
#else
	#define SCOPED_MUTEX(Name)
#endif
