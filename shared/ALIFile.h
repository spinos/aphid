#pragma once

#include <Alembic/Abc/IObject.h>
#include <Alembic/Abc/IArchive.h>

using namespace Alembic::Abc;

class ALITransform;
class ALIMesh;

class ALIFile
{
public:
    ALIFile();
    ~ALIFile();
    
    bool loadAbc(const char *filename);
    
    unsigned numTransform() const;
	unsigned numMesh() const;
	
	ALITransform *getTransform(unsigned idx);
	ALIMesh *getMesh(unsigned idx);
	
private:
    void ALIFile::listObject( IObject iObj, std::string iIndent );
    void flush();
    
    IArchive m_archive;
    std::vector<ALITransform *> m_transform;
	std::vector<ALIMesh *> m_mesh;
};
