#include <memory>

// This define forces inclusion of numpy symbols only in the Hydra_module.cpp file
#define NUMPY_IMPORT_MODULE
#include "ScriptCmds/ScriptIncludes.h"
#include "ScriptCmds/HydraConfig.h"

HYDRA_CONFIG_MODULE();

// Make this a unique pointer just in case init can be run more than once
static std::unique_ptr<PyMethodDef[]> Hydra_methods = nullptr;
static std::unique_ptr<std::string[]> Hydra_docstrs = nullptr;

static struct PyModuleDef Hydra_moduledef =
{
	PyModuleDef_HEAD_INIT,
	"Hydra",
	PyDoc_STR("Python wrappers for the Hydra Image Processing Library."),
	-1,
	nullptr
};


// Main python module initialization entry point
MODULE_INIT_FUNC(Hydra)
{
	ScriptCommand::CommandList cmds = ScriptCommand::commands();

	Hydra_methods = std::unique_ptr<PyMethodDef[]>(new PyMethodDef[cmds.size()+1]);
	Hydra_docstrs = std::unique_ptr<std::string[]>(new std::string[cmds.size()]);

	ScriptCommand::CommandList::const_iterator it = cmds.cbegin();
	for ( int i=0; it != cmds.cend(); ++it, ++i )
	{
		const ScriptCommand::FuncPtrs& cmdFuncs = it->second;
		const char* cmdName = it->first.c_str();

		Hydra_docstrs[i] = cmdFuncs.help();

		Hydra_methods[i] = {cmdName,cmdFuncs.dispatch,
					METH_VARARGS, PyDoc_STR(Hydra_docstrs[i].c_str()) };
	}

	// Methods list must end with null element
	Hydra_methods[cmds.size()] ={ nullptr, nullptr, 0, nullptr };


	Hydra_moduledef.m_methods = Hydra_methods.get();

	PyObject* hydra_module = PyModule_Create(&Hydra_moduledef);
	if ( !hydra_module )
		return nullptr;

	// Support for numpy arrays
	import_array();

	return hydra_module;
}
