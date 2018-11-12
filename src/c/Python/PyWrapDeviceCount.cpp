#include "PyWrapCommand.h"
#include "../Cuda/CWrappers.h"


const char PyWrapDeviceCount::docString[] = "NumCudaDevices, MemoryStats = HIP.DeviceCount()\n\n"\
	"This will return the number of Cuda devices available, and their memory.\n"\
	"\tNumCudaDevices -- this is the number of Cuda devices available.\n"\
	"\tMemoryStats -- this is an array of structures where each entry corresponds to a Cuda device.\n"\
	"The memory structure contains the total memory on the device and the memory available for a Cuda call.\n";

PyObject* PyWrapDeviceCount::execute(PyObject* self, PyObject* args)
{
	if ( !PyArg_ParseTuple(args, "") )
		return nullptr;

	std::size_t* memStats;
	int numDevices = memoryStats(&memStats);

	PyObject* py_mem = PyList_New(numDevices);
	for ( int i=0; i < numDevices; ++i )
	{
		PyObject* py_struct = Py_BuildValue("{sksk}", "total", memStats[i*2], "available", memStats[i*2+1]);
		PyList_SetItem(py_mem, i, py_struct);
	}

	return Py_BuildValue("(kN)", numDevices, py_mem);
}
