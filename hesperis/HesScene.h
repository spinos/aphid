#ifndef HES_SCENE_H
#define HES_SCENE_H

#include <string>
#include <vector>

namespace aphid {

class ATriangleMeshGroup;
class BoundingBox;

class HesScene {

typedef std::vector<ATriangleMeshGroup * > MeshVecTyp;
	MeshVecTyp m_meshes;
	
public:
    enum CameraOperation {
        opUnknown = 0,
        opFrameAll = 1
    };
    
    HesScene();
    virtual ~HesScene();
    
    bool load(const std::string& fileName);
	void close();
	const int numMeshes() const;
	const ATriangleMeshGroup* mesh(const int& i) const;
    
   const BoundingBox calculateBBox() const;
    
protected:
	void loadMesh(const std::string& mshName);
	
private:
};

}
#endif
