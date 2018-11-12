#ifndef CMD_LINE_PRGS_H
#define CMD_LINE_PRGS_H

#include "Defines.h"

#include <string>
#include <iostream>
#include <math.h>
#include <stdio.h>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#include <ctime>
#endif


/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
* windows and linux. */

uint64 GetTimeMs64()
{
#ifdef _WIN32
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	* to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64 ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

	return ret;
#else
	/* Linux */
	struct timeval tv;

	gettimeofday(&tv, NULL);

	uint64 ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;

	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);

	return ret;
#endif
}

char* PrintTime(std::size_t timeInMS)
{
	char* buff = new char[256];
	double hr = floor(timeInMS / 3.6e+6);
	double tmNew = timeInMS - hr * 3.6e+6;
	double mn = floor(tmNew / 6.0e4);
	tmNew = tmNew - mn * 6.0e4;
	double sc = tmNew / 1000.0f;

	sprintf(buff, "%02dh:%02dm:%05.2fs", (unsigned int)hr, (unsigned int)mn, sc);

	return buff;
}

class CmdLineProgress
{
public:
	CmdLineProgress(unsigned int numIterations)
	{
		defaults();
		iterations = numIterations;
		firstTime = GetTimeMs64();
	}

	CmdLineProgress(unsigned int numIterations, bool overwrite)
	{
		defaults();
		iterations = numIterations;
		firstTime = GetTimeMs64();
		useBackspace = overwrite;
	}

	CmdLineProgress(unsigned int numIterations, bool overwrite, std::string title)
	{
		defaults();
		iterations = numIterations;
		firstTime = GetTimeMs64();
		useBackspace = overwrite;
		titleText = title;		
	}

	~CmdLineProgress()
	{	
		defaults();
	}

	void print(unsigned int curIteration)
	{
		uint64 cur = GetTimeMs64();

		double prcntDone = (double)curIteration / (double)iterations;
		uint64 elapsedTime = cur - firstTime;
		double totalTime = elapsedTime / prcntDone;
		double finishTime = firstTime + totalTime;
		uint64 timeLeft = (uint64)(floor(finishTime - cur));
		char* timeLeftStr = PrintTime(timeLeft);

		char printString[256];
		if (!titleText.empty())
			sprintf(printString, "%s: %5.2f%%%% est. %s", titleText.c_str(), prcntDone * 100, timeLeftStr);
		else
			sprintf(printString, "%5.2f%%%% est. %s", prcntDone * 100, timeLeftStr);

		delete[] timeLeftStr;

		// This is a hack to find the length of the string
		std::string prntStr = printString;
		
		printBackspaces();
		printf("%s", prntStr.c_str());
		std::cout.flush();

		if (useBackspace)
			backspaces = prntStr.size();
	}

	void clear(bool printTotalTime=false)
	{
		printBackspaces();

		if (printTotalTime)
		{
			uint64 totalTime = GetTimeMs64() - firstTime;
			char* timeString = PrintTime(totalTime);
			
			if (!titleText.empty())
				printf("%s took: %s\n", titleText.c_str(), timeString);
			else
				printf("Total: %s\n", timeString);

			std::cout.flush();

			delete[] timeString;
		}

		defaults();
	}

private:
	unsigned int backspaces;
	uint64 firstTime;
	unsigned int iterations;
	bool useBackspace;
	std::string titleText;

	// We need the number of iterations for this to work properly
	CmdLineProgress(){}

	void defaults()
	{
		backspaces = 0;
		firstTime = 0;
		iterations = 0;
		useBackspace = false;
		titleText.clear();
	}

	void printBackspaces()
	{
		if (useBackspace)
		{
			for (unsigned int i = 0; i < backspaces; ++i)
			{
				printf("\b \b");
			}
			std::cout.flush();
		}
	}
};

#endif
