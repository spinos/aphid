#include <HFile.h>
#include <AFrameRange.h>
#include <string>
class ATriangleMesh;
class H5FileIn : public HFile, public AFrameRange {
public:
    H5FileIn();
    H5FileIn(const char * name);
    
    ATriangleMesh * findBakedMesh(std::string & name);
    
    bool isFrameBegin() const;
    bool isFrameEnd() const;
    void frameBegin();
    void nextFrame();
    
    bool readFrame(Vector3F * dst, unsigned nv);
    const int currentFrame() const;
protected:


private:
    std::string m_bakeName;
    int m_currentFrame;
};
