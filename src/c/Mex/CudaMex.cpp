// Defining DLL_EXPORT_SYM forces export of mexFunction
#if defined(_MSC_VER)
  #define DLL_EXPORT_SYM __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define DLL_EXPORT_SYM __attribute__((visibility("default")))
#endif

#include "ScriptCmds/ScriptIncludes.h"
#include "ScriptCmds/HydraConfig.h"

HYDRA_CONFIG_MODULE();

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Require a string as command input.
	if ( nrhs < 1 || !mxIsChar(prhs[0]) )
	{
		ScriptCommand::printUsage();
		Script::errorMsg("First argument must be a command string");
	}

	Script::mx_unique_ptr<char> strPtr = Script::make_mx_unique(mxArrayToUTF8String(prhs[0]));
	if ( !strPtr )
		Script::errorMsg("Could not read command string");

	// TODO: Make command matching case-insensitive
	const ScriptCommand::FuncPtrs* funcs = ScriptCommand::findCommand(strPtr.get());
	if ( !funcs )
	{
		ScriptCommand::printUsage();
		Script::errorMsg("Invalid command: %s", strPtr.get());
	}

	// Remove command from arguments passed to dispatch
	int cmdNRHS = nrhs-1;
	const mxArray** cmdPRHS = &prhs[1];

	// Dispatch to script command handler
	funcs->dispatch(nlhs,plhs, cmdNRHS, cmdPRHS);
}
