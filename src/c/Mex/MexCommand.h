#pragma once
#include "../Cuda/Vec.h"
#include "../Cuda/ImageDimensions.cuh"

#include "MexIncludes.h"

#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <exception>


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

		usageStr += offset + mexName + "(command, ...)\n\n";
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

		usageStr += "\nUse " + mexName + "('help',command) for detailed command info.\n";

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

		usageString += mexName;
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

    static Vec<std::size_t> MexCommand::FillKernel(const mxArray* matKernelIn, float** kernel);

private:
	static std::string mexName;

	static const std::size_t m_numCommands;
	static MexCommand* const m_commands[];

	std::string m_cmdString;
};


#include "MexWrapDef.h"
#include "../WrapCmds/CommandList.h"
