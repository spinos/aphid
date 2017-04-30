#pragma once
#include <APlaybackFile.h>
#include <string>
class ATriangleMesh;
class H5FileIn : public APlaybackFile {
public:
    H5FileIn();
    H5FileIn(const char * name);
    
    ATriangleMesh * findBakedMesh(std::string & name);
    bool readFrame(Vector3F * dst, unsigned nv);
protected:


private:
    std::string m_bakeName;
};
