#include <iostream>
#include <geom/SuperQuadricGlyph.h>
#include <topo/TriangleMeshClique.h>

using namespace aphid;

int main(int argc, char *argv[])
{
    std::cout<<"\n test clique ";
	SuperQuadricGlyph glyph(10);
	std::cout<<"\n n site "<<glyph.numTriangles();
	TriangleMeshClique clique(&glyph);
	clique.findClique(99);
	
	std::vector<int> cinds;
	clique.getCliqueSiteIndices(cinds);
	
	std::vector<int>::const_iterator it = cinds.begin();
	for(;it!=cinds.end();++it) {
		std::cout<<" "<<*it;
	}
	
	std::cout<<"\n clique size "<<cinds.size();
	
    return 1;
}
