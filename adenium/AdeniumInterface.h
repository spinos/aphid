#ifndef ADENIUMINTERFACE_H
#define ADENIUMINTERFACE_H
#include <string>
#include <H5FileIn.h>
class AdeniumWorld;
class AdeniumInterface {
public:
    AdeniumInterface();
    virtual ~AdeniumInterface();
    void create(AdeniumWorld * world);
	void changeMaxDisplayLevel(AdeniumWorld * world, int x);
    static bool LoadBake(AdeniumWorld * world, const std::string & name);
    static bool ReadBakeFrame(AdeniumWorld * world);
    static std::string FileName;
    static H5FileIn BakeFile;
protected:
    bool readTriangleMeshFromFile(AdeniumWorld * world);
    bool readTetrahedronMeshFromFile(AdeniumWorld * world);
};
#endif        //  #ifndef ADENIUMINTERFACE_H

