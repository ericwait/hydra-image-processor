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

	static void setupImagePointers(const mxArray* imageIn,unsigned char** image,Vec<size_t>* dims,mxArray** argOut=NULL,unsigned char** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,unsigned short** image,Vec<size_t>* dims,mxArray** argOut=NULL,unsigned short** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,short** image,Vec<size_t>* dims,mxArray** argOut=NULL,short** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,unsigned int** image,Vec<size_t>* dims,mxArray** argOut=NULL,unsigned int** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,int** image,Vec<size_t>* dims,mxArray** argOut=NULL,int** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,float** image,Vec<size_t>* dims,mxArray** argOut=NULL,float** imageOut=NULL);
	static void setupImagePointers(const mxArray* imageIn,double** image,Vec<size_t>* dims,mxArray** argOut=NULL,double** imageOut=NULL);

private:
	static const size_t m_numCommands;
	static MexCommand* const m_commands[];

	std::string m_cmdString;
};


#include "CommandList.h"
