#include <iostream>
#include <geom/SuperQuadricGlyph.h>
#include <topo/TriangleMeshClique.h>
#include <geom/PrincipalComponents.h>
#include <gpr/PCAFeature.h>
#include <gpr/PCASimilarity.h>

using namespace aphid;

int main(int argc, char *argv[])
{
    std::cout<<"\n test clique ";
	SuperQuadricGlyph glyph(10);
	glyph.computePositions(.75f, .5f);
	
	const int nv = glyph.numPoints();
	
	std::cout<<"\n total n site "<<glyph.numTriangles();
	TriangleMeshClique clique(&glyph);
	clique.findClique(553, glyph.numTriangles() );
	
	std::cout<<"\n first clique of n site "<<clique.numSites();
	
	sdb::Sequence<int> visited;
	clique.getCliqueSiteIndices(visited);
	
	std::vector<Vector3F > vertps;
	clique.getCliqueVertexPositions(vertps);
	std::cout<<"\n n vert "<<vertps.size();
	
	PCASimilarity<float, PCAFeature<float> > features;
	features.begin(vertps);
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
    AOrientedBox obox = obpca.analyze(vertps, vertps.size() );
	std::cout<<"\n obox "<<obox;
	
	vertps.clear();
	
	const int nsthre = clique.numSites() * 2;
	std::cout<<"\n loop through all sites to find clique by size "<<nsthre;
	
	const int nf = glyph.numTriangles();
	for(int i=0;i<nf;++i) {
		if(visited.findKey(i) ) {
/// already assigned to a clique
			continue;
		}
		
		clique.findClique(i, nsthre);
		
		std::cout<<"\n found a clique of n site "<<clique.numSites();
		clique.getCliqueSiteIndices(visited);
		
		clique.getCliqueVertexPositions(vertps);
		std::cout<<"\n n vert "<<vertps.size();
		features.select(vertps);
		
		vertps.clear();
		
	}
	
	std::cout<<"\n n features "<<features.numFeatures();
	features.computeSimilarity();
	
    return 1;
}
