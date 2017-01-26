#pragma once
#include "Vec.h"

#include "mex.h"
#include "windows.h"
#undef min
#undef max

#include <vector>
#include <string>
#include <algorithm>
#include <exception>

// Static class for holding some module information
class ModuleInfo
{
public:
	static void setModuleHandle(HMODULE handle)
	{
		hModule = handle;
	}

	static void initModuleInfo()
	{
		if(name.length() == 0)
			name = initName();
	}

	static const std::string& getName()
	{
		return name;
	}

private:
	static std::string initName()
	{
		if(hModule == NULL)
			return "";

		char pathBuffer[1024];
		DWORD result = GetModuleFileName((HMODULE)hModule,pathBuffer,1024);
		if(FAILED(result))
			return "";

		std::string path(pathBuffer);

		size_t startOffset = path.find_last_of('\\') + 1;
		path = path.substr(startOffset);

		size_t endOffset = path.find_last_of('.');

		return path.substr(0,endOffset);
	}

private:
	static std::string name;
	static HMODULE hModule;
};

// Abstract base class for mex commands
class MexCommand
{
public:
	virtual std::string check(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const = 0;
	virtual void execute(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) const = 0;

	virtual void usage(std::vector<std::string>& outArgs,std::vector<std::string>& inArgs) const = 0;
	virtual void help(std::vector<std::string>& helpLines) const = 0;
	const std::string& command() const
	{
		return m_cmdString;
	}

public:
	// Runs through any registered commands.
	static void run(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[])
	{
		ModuleInfo::initModuleInfo();

		// Require a string as command input.
		if(nrhs < 1 || !mxIsChar(prhs[0]))
			mexErrMsgTxt(MexCommand::printUsageList().c_str());

		char cmdBuffer[256];
		int cmdLen = mxGetString(prhs[0],cmdBuffer,256);

		MexCommand* mexCmd = MexCommand::getCommand(cmdBuffer);
		if(mexCmd == NULL)
			mexErrMsgTxt(MexCommand::printUsageList().c_str());

		int cmdNRHS = nrhs-1;
		const mxArray** cmdPRHS = &prhs[1];

		std::string errMsg = mexCmd->check(nlhs,plhs,cmdNRHS,cmdPRHS);
		if(errMsg.length()!=0)
		{
			mexPrintf(MexCommand::printCommandHelp(mexCmd).c_str());
			mexErrMsgTxt(errMsg.c_str());
		}

		try
		{
			mexCmd->execute(nlhs,plhs,cmdNRHS,cmdPRHS);
		} catch(const std::runtime_error& err)
		{
			std::string errString(mexCmd->m_cmdString);
			errString += ": " + std::string(err.what());

			mexErrMsgTxt(errString.c_str());
		}
	}

	static std::string printDocumentation()
	{
		std::string offset("   ");
		std::string usageStr;

		ModuleInfo::initModuleInfo();

		usageStr += offset + ModuleInfo::getName() + "(command, ...)\n\n";
		usageStr += offset + "Command List:\n";

		for(int i=0; i < m_numCommands; ++i)
			usageStr += printCommandHelp(m_commands[i],offset) + "\n";

		return usageStr;
	}

	static std::vector<MexCommand*> getCommandList()
	{
		std::vector<MexCommand*> commandList(m_numCommands);
		for(int i = 0; i < m_numCommands; ++i)
			commandList[i] = m_commands[i];

		return commandList;
	}

protected:
	static std::string printUsageList()
	{
		std::string usageStr = "Usage:\n";

		for(int i=0; i < m_numCommands; ++i)
			usageStr += "  " + buildUsageString(m_commands[i]) + "\n";

		usageStr += "\nUse " + ModuleInfo::getName() + "('help',command) for detailed command info.\n";

		return usageStr;
	}

	static std::string printCommandHelp(MexCommand* mexCmd,std::string offset = "")
	{
		std::string helpStr = offset + buildUsageString(mexCmd) + "\n";
		std::vector<std::string> helpLines;

		mexCmd->help(helpLines);

		for(int i=0; i < helpLines.size(); ++i)
			helpStr += offset + "  " + helpLines[i] + "\n";

		return helpStr;
	}

	static MexCommand* getCommand(const std::string& commandString)
	{
		std::string chkCommand(commandString);

		for(int i=0; i < m_numCommands; ++i)
		{
			if(m_commands[i]->m_cmdString != chkCommand)
				continue;

			return m_commands[i];
		}

		return NULL;
	}

protected:
	MexCommand(const char* command) : m_cmdString(command) {}

	static std::string buildUsageString(const MexCommand* mexCmd)
	{
		std::string usageString;

		std::vector<std::string> inputs;
		std::vector<std::string> outputs;

		mexCmd->usage(outputs,inputs);
		if(outputs.size() > 0)
		{
			usageString += "[";
			for(int i = 0; i < outputs.size()-1; ++i)
				usageString += outputs[i] + ",";

			usageString += outputs[outputs.size()-1] + "] = ";
		}

		usageString += ModuleInfo::getName();
		usageString += "(";
		usageString += "'" + mexCmd->m_cmdString + "'";

		if(inputs.size() > 0)
		{
			usageString += ", ";
			for(int i = 0; i < inputs.size() - 1; ++i)
				usageString += inputs[i] + ", ";

			usageString += inputs[inputs.size() - 1];
		}
		usageString += ")";

		return usageString;
	}

	// Helper function for MexCommands class
	static size_t countDims(Vec<size_t> dims);
	static void setupDims(const mxArray* im, Vec<size_t>* dims);

	// Simple template-specialization map for C++ to mex types
	template <typename T> struct TypeMap		{static const mxClassID mxType;};
	template <> struct TypeMap<char>			{static const mxClassID mxType = mxINT8_CLASS;};
	template <> struct TypeMap<short>			{static const mxClassID mxType = mxINT16_CLASS;};
	template <> struct TypeMap<int>				{static const mxClassID mxType = mxINT32_CLASS;};
	template <> struct TypeMap<unsigned char>	{static const mxClassID mxType = mxUINT8_CLASS;};
	template <> struct TypeMap<unsigned short>	{static const mxClassID mxType = mxUINT16_CLASS;};
	template <> struct TypeMap<unsigned int>	{static const mxClassID mxType = mxUINT32_CLASS;};
	template <> struct TypeMap<float>			{static const mxClassID mxType = mxSINGLE_CLASS;};
	template <> struct TypeMap<double>			{static const mxClassID mxType = mxDOUBLE_CLASS;};

	// General array creation method
	template <typename T>
	static mxArray* createArray(mwSize ndim, const mwSize* dims)
	{
		return mxCreateNumericArray(ndim, dims, TypeMap<T>::mxType, mxREAL);
	}

	// Logical array creation specialziation
	template <>
	static mxArray* createArray<bool>(mwSize ndim, const mwSize* dims)
	{
		return mxCreateLogicalArray(ndim, dims);
	}

	template <typename T>
	static void setupImagePointers(const mxArray* imageIn, T** image, Vec<size_t>* dims, mxArray** argOut = NULL, T** imageOut = NULL)
	{
		setupInputPointers(imageIn, dims, image);
		if (argOut!=NULL && imageOut!=NULL)
			setupOutputPointers(argOut, *dims, imageOut);
	}

	template <typename T>
	static void setupInputPointers(const mxArray* imageIn, Vec<size_t>* pDims, T** image)
	{
		setupDims(imageIn, pDims);
		*image = (T*)mxGetData(imageIn);
	}

	template <typename T>
	static void setupOutputPointers(mxArray** imageOut, Vec<size_t> dims, T** image)
	{
		size_t numDims = countDims(dims);

		*imageOut = createArray<T>(numDims, dims.e);
		*image = (T*)mxGetData(*imageOut);

		memset(*image, 0, sizeof(T)*dims.product());
	}


    static Vec<size_t> MexCommand::FillKernel(const mxArray* matKernelIn, float** kernel);

private:
	static const size_t m_numCommands;
	static MexCommand* const m_commands[];

	std::string m_cmdString;
};


#include "CommandList.h"
