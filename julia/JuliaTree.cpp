#include "JuliaTree.h"
#include <HInnerGrid.h>
#include <HWorldGrid.h>
#include <ConvexShape.h>

using namespace aphid;

namespace jul {

JuliaTree::JuliaTree(Parameter * param) 
{
    sdb::TreeNode::MaxNumKeysPerNode = 128;
	sdb::TreeNode::MinNumKeysPerNode = 16;

	HObject::FileIO.open(param->outFileName().c_str(), HDocument::oReadAndWrite);
	
	std::vector<std::string > gridNames;
	HBase r("/");
	r.lsTypedChild<HVarGrid>(gridNames);
	
	if(gridNames.size() > 0) buildTree(gridNames[0]);
	else std::cout<<"\n found no grid";
	
	r.close();
	HObject::FileIO.close();
}

JuliaTree::~JuliaTree() {}

void JuliaTree::buildTree(const std::string & name)
{
    cvx::ShapeType vt = cvx::TUnknown;
    
    HVarGrid vg(name);
    vg.load();
    std::cout<<"\n value type ";
    switch(vg.valueType() ) {
        case cvx::TSphere :
        std::cout<<"sphere";
        vt = cvx::TSphere;
        break;
    default:
        std::cout<<"unsupported";
        break;
    }
    vg.close();
    
    if(vt == cvx::TSphere) buildSphere(name);
}

void JuliaTree::buildSphere(const std::string & name)
{
    
}

}
