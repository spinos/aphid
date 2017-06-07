#ifndef HES_SCENE_H
#define HES_SCENE_H

#include <string>
#include <vector>

namespace aphid {

class ATriangleMeshGroup;

class HesScene {

typedef std::vector<ATriangleMeshGroup * > MeshVecTyp;
	MeshVecTyp m_meshes;
	
public:
    HesScene();
    virtual ~HesScene();
    
    bool load(const std::string& fileName);
	void close();
	const int numMeshes() const;
	const ATriangleMeshGroup* mesh(const int& i) const;
    
protected:
	void loadMesh(const std::string& mshName);
	
private:
};

}
#endif
