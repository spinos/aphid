#include "JuliaTree.h"
#include <HInnerGrid.h>
#include <HWorldGrid.h>
#include <KdEngine.h>
#include <NTreeIO.h>

using namespace aphid;

namespace jul {

JuliaTree::JuliaTree(Parameter * param) 
{
	NTreeIO hio;
	hio.begin(param->outFileName(), HDocument::oReadAndWrite );
	
	std::string gridName;
	if(hio.findGrid(gridName))
		buildTree(gridName);
		
	hio.end();
}

JuliaTree::~JuliaTree() {}

void JuliaTree::buildTree(const std::string & name)
{
    NTreeIO hio;
	cvx::ShapeType vt = hio.gridValueType(name);
    
    if(vt == cvx::TSphere) buildSphere(name);
}

void JuliaTree::buildSphere(const std::string & name)
{
    sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 4, 1024 >, cvx::Sphere > grd(name);
    grd.load();
    
    const float h = grd.gridSize();
    const float e = h * .49995f;
    sdb::VectorArray<cvx::Cube> cs;
    cvx::Cube c;
    grd.begin();
    while(!grd.end() ) {
        c.set(grd.coordToCellCenter(grd.key() ), e);
        cs.insert(c);
        grd.next();   
    }
    
    HNTree<cvx::Cube, KdNode4 > tree( boost::str(boost::format("%1%/tree") % name ) );
    KdEngine<cvx::Cube> engine;
    TreeProperty::BuildProfile bf;
    bf._maxLeafPrims = 5;
    
    engine.buildTree(&tree, &cs, grd.boundingBox(), &bf);
	
	tree.save();
	tree.close();
	grd.close();
    
}

}
