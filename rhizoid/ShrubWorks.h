/*
 *  ShrubWorks.h
 *  
 *
 *  Created by jian zhang on 12/26/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <vector>

namespace aphid {

class BoundingBox;

template<typename T>
class DenseMatrix;

template<typename T1, int N>
class PCAFeature;

template<typename T1, typename T2>
class PCASimilarity;

class ShrubWorks {

public:
	ShrubWorks();
	virtual ~ShrubWorks();
	
protected:
	MStatus creatShrub();
	
private:
	int getGroupMeshVertices(DenseMatrix<float> * vertices,
					BoundingBox & bbox, 
					const MDagPath & path) const;
	void countMeshNv(int & nv,
					const MDagPath & meshPath) const;
	void getMeshVertices(DenseMatrix<float> * vertices, 
					int iRow, 
					BoundingBox & bbox, 
					const MDagPath & meshPath, 
					const MDagPath & transformPath) const;
					
typedef PCASimilarity<float, PCAFeature<float, 3> > SimilarityType;

	bool findSimilar(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices) const;
	void addSimilarity(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices) const;
	void clearSimilarity(std::vector<SimilarityType * > & similarities) const;
	void separateFeatures(std::vector<SimilarityType * > & similarities) const;
	
};

}