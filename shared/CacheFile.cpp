#include "CacheFile.h"

CacheFile::CacheFile() : BaseFile() {}
CacheFile::CacheFile(const char * name) : BaseFile(name) {}

bool CacheFile::open(const std::string & fileName)
{
    if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}

	return BaseFile::open(fileName);
}

bool CacheFile::save()
{
    return true;
}

bool CacheFile::close()
{
    return true;
}
