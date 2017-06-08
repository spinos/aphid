#include "HesScene.h"
#include <h5/HesperisFile.h>
#include <geom/ATriangleMeshGroup.h>
#include <foundation/SHelper.h>

namespace aphid {

HesScene::HesScene()
{}

HesScene::~HesScene()
{}

bool HesScene::load(const std::string& fileName)
{
	close();
    HesperisFile* hesDoc = new HesperisFile;
    hesDoc->setOpenMode(HDocument::oReadOnly);
    
    std::string hesname(fileName);
	SHelper::changeFilenameExtension(hesname, "hes");
    bool stat = hesDoc->open(hesname);
    if(!stat) {
        return false;
    }
    std::vector<std::string > mshNames;
    HWorld grpWorld;
    HesperisFile::LsNames<HTriangleMeshGroup>(mshNames, &grpWorld);
    grpWorld.close();
    
    const int nmsh = mshNames.size();
	for(int i=0;i<nmsh;++i) {
		loadMesh(mshNames[i]);
	}
	
	hesDoc->close();
	delete hesDoc;
    return true;
}

void HesScene::close()
{
	MeshVecTyp::iterator it = m_meshes.begin();
	for(;it!=m_meshes.end();++it) {
		delete *it;
	}
	m_meshes.clear();
}

void HesScene::loadMesh(const std::string& mshName)
{
	std::cout<<"\n load mesh "<<mshName;
	HTriangleMeshGroup grp(mshName);
	ATriangleMeshGroup* tri = new ATriangleMeshGroup;
	grp.load(tri);
	m_meshes.push_back(tri);
	grp.close();
}

const int HesScene::numMeshes() const
{ return m_meshes.size(); }

const ATriangleMeshGroup* HesScene::mesh(const int& i) const
{ return m_meshes[i]; }

const BoundingBox HesScene::calculateBBox() const
{
    BoundingBox rb;
    MeshVecTyp::const_iterator it = m_meshes.begin();
	for(;it!=m_meshes.end();++it) {
		rb.expandBy((*it)->calculateGeomBBox() );
	}
    return rb;
}

}

