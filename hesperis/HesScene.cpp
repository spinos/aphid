#include "HesScene.h"
#include <h5/HesperisFile.h>
#include <foundation/SHelper.h>

namespace aphid {

HesScene::HesScene()
{}

HesScene::~HesScene()
{}

bool HesScene::load(const std::string& fileName)
{
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
    
    SHelper::PrintVectorStr(mshNames);
    return true;
}

}

