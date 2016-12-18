#include <iostream>
#include <geom/SuperQuadricGlyph.h>
#include <topo/TriangleMeshClique.h>
#include <geom/PrincipalComponents.h>

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
	
	std::vector<Vector3F > vertps;
	clique.getCliqueVertexPositions(vertps);
	std::cout<<"\n n vert "<<vertps.size();
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
    AOrientedBox obox = obpca.analyze(vertps, vertps.size() );
	std::cout<<"\n obox "<<obox;
	
	std::vector<Vector3F > localps;
	const int n = vertps.size();
	for(int i=0;i<n;++i) {
		localps.push_back(vertps[i]);
	}
	obox.projectToLocalUnit<std::vector<Vector3F > >(localps, n);
	
	vertps.clear();
    return 1;
}
