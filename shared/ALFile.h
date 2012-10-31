#pragma once
#include <Alembic/Abc/All.h>
#include <vector>
using namespace Alembic::Abc;

class ALFile
{
public:
    ALFile();
    ~ALFile();
    
    void openAbc(const char *filename);
	void addTimeSampling();
    OObject root();
    char findObject(const std::vector<std::string> &path, OObject &dest);
	char findObject(const std::string &fullPathName, OObject &dest);
    char findChildByName(OObject &parent, OObject &child, const std::string &name);
	char findParentOf(const std::string &fullPathName, OObject &dest);
	
	static void splitNames(const std::string &fullPath, std::vector<std::string> &paths);

private:
	
    OArchive m_archive;
};
