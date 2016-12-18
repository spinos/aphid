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
	clique.findClique(53, glyph.numTriangles() );
	
	std::cout<<"\n clique size "<<clique.numSites();
	
	sdb::Sequence<int> visited;
	clique.getCliqueSiteIndices(visited);
	
    return 1;
}
