#pragma once
// HydraConfig - Provides a simple environment-variable based config loader to allow some
//               configuration of the Hyrda library (e.g. enabling process-level mutex)

#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

class HydraConfig
{
public:
	static bool validConfig()
	{
		return m_staticInst;
	}

	static bool useProcessMutex()
	{
		if (!m_staticInst)
			return false;

		return m_staticInst->bUseProcessMutex;
	}

private:
	HydraConfig()
	{
		defaultConfig(this);
	}

	static void defaultConfig(HydraConfig* pInst)
	{
		pInst->bUseProcessMutex = false;
	}

	// TODO: Use env-variable to find a config file instead of env-variables for direct configuration
	static void loadConfig(HydraConfig* pInst)
	{
		char* envUPM = std::getenv("HYDRA_ENABLE_MUTEX");
		if ( envUPM )
		{
			std::string envStr(envUPM);
			std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::toupper);

			if ( envStr == "TRUE" || envStr == "1" )
				pInst->bUseProcessMutex = true;
		}
	}

	static HydraConfig* initConfig()
	{
		HydraConfig* pInst = new HydraConfig();

		HydraConfig::loadConfig(pInst);
		return pInst;
	}

private:
	bool bUseProcessMutex;

	static HydraConfig* m_staticInst;
};

#define HYDRA_CONFIG_MODULE() HydraConfig* HydraConfig::m_staticInst = HydraConfig::initConfig()
