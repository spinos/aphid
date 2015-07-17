#include <AllHdf.h>
#include <AFrameRange.h>
#include <string>
class ATriangleMesh;
class H5FileIn : public HFile, public AFrameRange {
public:
    H5FileIn();
    H5FileIn(const char * name);
    
    ATriangleMesh * findBakedMesh(std::string & name);
protected:


private:

};
