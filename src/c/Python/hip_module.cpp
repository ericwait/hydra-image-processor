#define NUMPY_IMPORT_MODULE
#include "PyIncludes.h"

// Defined in PyWrapCommand.cpp
extern struct PyMethodDef hip_methods[];

static struct PyModuleDef hip_moduledef =
{
	PyModuleDef_HEAD_INIT,
	"HIP",
	PyDoc_STR("Python wrappers for the Hydra Image Processing Library."),
	-1,
	hip_methods
};


// Main module initialization entry point
MODULE_INIT_FUNC(HIP)
{
	PyObject* hip_module = PyModule_Create(&hip_moduledef);
	if ( !hip_module )
		return nullptr;

	// Support for numpy arrays
	import_array();

	return hip_module;
}
