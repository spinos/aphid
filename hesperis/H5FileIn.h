#include <HFile.h>
#include <AFrameRange.h>
#include <string>
class ATriangleMesh;
class H5FileIn : public HFile, public AFrameRange {
public:
    H5FileIn();
    H5FileIn(const char * name);
    
    ATriangleMesh * findBakedMesh(std::string & name);
    
    void frameBegin();
    bool frameEnd() const;
    void nextFrame();
    
    bool readFrame(Vector3F * dst, unsigned nv);
protected:


private:
    std::string m_bakeName;
    int m_currentFrame;
};
