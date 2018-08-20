#include "PyWrapCommand.h"

#include "../Cuda/CWrappers.h"
#include "../Cuda/CudaDeviceStats.h"


const char PyWrapDeviceStats::docString[] = "DeviceStatsArray = HIP.DeviceStats()\n\n"\
	"This will return the statistics of each Cuda capable device installed.\n"\
	"\tDeviceStatsArray -- this is an array of structs, one struct per device.\n"\
	"The struct has these fields: name, major, minor, constMem, sharedMem, totalMem, tccDriver, mpCount, threadsPerMP, warpSize, maxThreads.\n";


PyObject* PyWrapDeviceStats::execute(PyObject* self, PyObject* args)
{
	if ( !PyArg_ParseTuple(args, "") )
		return nullptr;

	DevStats* devStats;
	int numDevices = deviceStats(&devStats);

	PyObject* py_stats = PyList_New(numDevices);
	for ( int device = 0; device<numDevices; ++device )
	{
		DevStats& dev = devStats[device];

		PyObject* dict = PyDict_New();
		PyDict_SetItemString(dict, "name", PyStr_FromString(dev.name.c_str()));
		PyDict_SetItemString(dict, "major", PyLong_FromLong(dev.major));
		PyDict_SetItemString(dict, "minor", PyLong_FromLong(dev.minor));
		PyDict_SetItemString(dict, "constMem", PyLong_FromUnsignedLongLong(dev.constMem));
		PyDict_SetItemString(dict, "sharedMem", PyLong_FromUnsignedLongLong(dev.sharedMem));
		PyDict_SetItemString(dict, "totalMem", PyLong_FromUnsignedLongLong(dev.totalMem));
		PyDict_SetItemString(dict, "tccDriver", PyBool_FromLong(dev.tccDriver));
		PyDict_SetItemString(dict, "mpCount", PyLong_FromLong(dev.mpCount));
		PyDict_SetItemString(dict, "threadsPerMP", PyLong_FromLong(dev.threadsPerMP));
		PyDict_SetItemString(dict, "warpSize", PyLong_FromLong(dev.warpSize));
		PyDict_SetItemString(dict, "maxThreads", PyLong_FromLong(dev.maxThreads));

		PyList_SetItem(py_stats, device, dict);
	}

	delete[] devStats;

	return py_stats;
}
