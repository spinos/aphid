#pragma once
#include <Alembic/Abc/All.h>

class ALFile
{
public:
    ALFile();
    ~ALFile();
    
    void openAbc(const char *filename);
    Alembic::Abc::OObject root();
    char object(const std::string &objectPath, Alembic::Abc::OObject &dest);
    char findChildByName(Alembic::Abc::OObject &parent, Alembic::Abc::OObject &child, const std::string &name);
private: 
    Alembic::Abc::OArchive m_archive;
};
