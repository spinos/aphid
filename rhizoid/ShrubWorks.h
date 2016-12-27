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
#include <maya/MDagPathArray.h>
#include <vector>
#include <map>

namespace aphid {

namespace sdb {

template<typename T1, typename T2>
class Couple;

}

struct Int2;

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
					int & iRow, 
					BoundingBox & bbox, 
					const MDagPath & meshPath, 
					const MDagPath & transformPath) const;
		
/// groupId and similarity
typedef sdb::Couple<int, PCASimilarity<float, PCAFeature<float, 3> > > SimilarityType;

	bool findSimilar(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices) const;
	void addSimilarity(std::vector<SimilarityType * > & similarities, 
					const DenseMatrix<float> & vertices,
					const int & groupi) const;
	void clearSimilarity(std::vector<SimilarityType * > & similarities) const;
	
	void addSimilarities(std::vector<SimilarityType * > & similarities,
					BoundingBox & totalBox,
					const MDagPathArray & paths) const;

/// similarity<<8 | example as key
/// (group, global_ind) as value
typedef std::map<int, Int2 > FeatureExampleMap;

/// a relative to b
	void scaleSpace(DenseMatrix<float> & space,
					const float * a,
					const float * b) const;
	
/// serialize global_ind
	int countExamples(const std::vector<SimilarityType * > & similarities,
					FeatureExampleMap & exampleGroupInd) const;
	void addInstances(const std::vector<SimilarityType * > & similarities,
					 FeatureExampleMap & exampleGroupInd) const;
	MObject createShrubViz(const BoundingBox & bbox) const;
	
};

}