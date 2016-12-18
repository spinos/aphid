#include <iostream>
#include <geom/SuperQuadricGlyph.h>
#include <topo/TriangleMeshClique.h>
#include <geom/PrincipalComponents.h>

using namespace aphid;

int main(int argc, char *argv[])
{
    std::cout<<"\n test clique ";
	SuperQuadricGlyph glyph(10);
	glyph.computePositions(1.5f, .5f);
	
	const int nv = glyph.numPoints();
	
	std::cout<<"\n n site "<<glyph.numTriangles();
	TriangleMeshClique clique(&glyph);
	clique.findClique(553, glyph.numTriangles() );
	
	std::cout<<"\n clique size "<<clique.numSites();
	
	sdb::Sequence<int> visited;
	clique.getCliqueSiteIndices(visited);
	
	std::vector<Vector3F > vertps;
	clique.getCliqueVertexPositions(vertps);
	std::cout<<"\n n vert "<<vertps.size();
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
    AOrientedBox obox = obpca.analyze(vertps, vertps.size() );
	std::cout<<"\n obox "<<obox;
	
	vertps.clear();
    return 1;
}
