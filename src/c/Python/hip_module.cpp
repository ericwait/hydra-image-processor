#include <memory>


#define NUMPY_IMPORT_MODULE
#include "PyIncludes.h"

#include "../WrapCmds/ScriptCommand.h"

// Make this a unique pointer just in case init can be run more than once
static std::unique_ptr<PyMethodDef[]> hip_methods = nullptr;
static std::unique_ptr<std::string[]> hip_docstrs = nullptr;

static struct PyModuleDef hip_moduledef =
{
	PyModuleDef_HEAD_INIT,
	"HIP",
	PyDoc_STR("Python wrappers for the Hydra Image Processing Library."),
	-1,
	nullptr
};


// Main module initialization entry point
MODULE_INIT_FUNC(HIP)
{
	ScriptCommand::CommandList cmds = ScriptCommand::commands();

	hip_methods = std::unique_ptr<PyMethodDef[]>(new PyMethodDef[cmds.size()+1]);
	hip_docstrs = std::unique_ptr<std::string[]>(new std::string[cmds.size()]);

	ScriptCommand::CommandList::const_iterator it = cmds.cbegin();
	for ( int i=0; it != cmds.cend(); ++it, ++i )
	{
		const ScriptCommand::FuncPtrs& cmdFuncs = it->second;
		const char* cmdName = it->first.c_str();

		hip_docstrs[i] = cmdFuncs.help();

		hip_methods[i] = {cmdName,cmdFuncs.dispatch,
					METH_VARARGS, PyDoc_STR(hip_docstrs[i].c_str()) };
	}

	// Methods list must end with null element
	hip_methods[cmds.size()] ={ nullptr, nullptr, 0, nullptr };


	hip_moduledef.m_methods = hip_methods.get();

	PyObject* hip_module = PyModule_Create(&hip_moduledef);
	if ( !hip_module )
		return nullptr;

	// Support for numpy arrays
	import_array();

	return hip_module;
}
