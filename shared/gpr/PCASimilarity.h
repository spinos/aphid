/*
 *  PCASimilarity.h
 *  
 *  similarity search via pca
 *
 *  Created by jian zhang on 12/20/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PCA_SIMILARITY_H
#define APH_PCA_SIMILARITY_H

#include <math/linearMath.h>

namespace aphid {

template<typename T, typename Tf>
class PCASimilarity {

	std::vector<Tf * > m_features;
	
public:
	PCASimilarity();
	virtual ~PCASimilarity();
	
/// set example by pnts, initialize search
	void begin(const std::vector<Vector3F > & pnts);
/// select a feature
	bool select(const std::vector<Vector3F > & pnts);
	
/// dim = 1 columnwise data
/// dim = 2 rowwise data
	void begin(const DenseMatrix<T> & pnts,
				const int & dim);
	bool select(const DenseMatrix<T> & pnts,
				const int & dim);
	
	void computeSimilarity();
	
/// n row of first feature
	int featureDim() const;
	
	int numFeatures() const;
	
	void getFeatureSpace(T * dst,
					const int & idx) const;
/// dim = 1 stored columnwise
/// dim = 2 stored rowwise
	void getFeaturePoints(DenseMatrix<T> & dst,
					const int & idx,
					const int & dim) const;
	void getFeatureBound(T * dst,
					const int & idx,
					const int & dim) const;
	
protected:

private:
	void clearFeatures();
	void addFeature(const DenseMatrix<T> & pnts,
				const int & dim);
				
};

template<typename T, typename Tf>
PCASimilarity<T, Tf>::PCASimilarity()
{}

template<typename T, typename Tf>
PCASimilarity<T, Tf>::~PCASimilarity()
{
	clearFeatures();
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::addFeature(const DenseMatrix<T> & pnts,
				const int & dim)
{
	if(dim==1) {
		const int np = pnts.numCols();
		Tf * af = new Tf(np);
		for(int i=0;i<np;++i) {
			af->setPnt(pnts.column(i), i);
		}
		m_features.push_back(af);
		
	} else {
		const int np = pnts.numRows();
		const int nvar = pnts.numCols();
		DenseVector<T> apnt(nvar);
		
		Tf * af = new Tf(np);
		for(int i=0;i<np;++i) {
			for(int j=0;j<nvar;++j) {
				apnt[j] = pnts.column(i)[j];
			}
			
			af->setPnt(apnt.v(), i);
		}
		m_features.push_back(af);
	}
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::begin(const DenseMatrix<T> & pnts,
				const int & dim)
{
	clearFeatures();
	
	if(dim==1) {
		if(pnts.numRows() != Tf::numVars() ) {
			throw " PCASimilarity cannot begin with wrong data dim col";
		}
		
	} else {
		if(pnts.numCols() != Tf::numVars() ) {
			throw " PCASimilarity cannot begin with wrong data dim row";
		}
	}
	
	addFeature(pnts, dim);
}

template<typename T, typename Tf>
bool PCASimilarity<T, Tf>::select(const DenseMatrix<T> & pnts,
				const int & dim)
{
	const int exnp = featureDim();
	if(exnp<1) {
		return false;
	}
	
	if(dim==1) {
		if(pnts.numRows() != Tf::numVars() ) {
			return false;
		}
		if(pnts.numCols() != exnp) {
			return false;
		}
		
	} else {
		if(pnts.numCols() != Tf::numVars() ) {
			return false;
		}
		if(pnts.numRows() != exnp) {
			return false;
		}
	}
	addFeature(pnts, dim);
	return true;
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::begin(const std::vector<Vector3F > & pnts)
{
	clearFeatures();
	
	const int np = pnts.size();
	Tf * af = new Tf(np);
	for(int i=0;i<np;++i) {
		af->setPnt((const T *)&pnts[i], i);
	}
	m_features.push_back(af);
}

template<typename T, typename Tf>
bool PCASimilarity<T, Tf>::select(const std::vector<Vector3F > & pnts)
{
	const int exnp = featureDim();
	if(exnp<1) {
		return false;
	}
	const int np = pnts.size();
	if(np < exnp/2 || np > (exnp + exnp/2) ) {
		return false;
	}
	
	Tf * af = new Tf(exnp);
	
	int unp = np;
	if(np < exnp) {
/// lack of points
		af->copy(*m_features[0]);
	} else if(np > exnp) {
/// more than enough points
		unp = exnp;
	}
	
	for(int i=0;i<unp;++i) {
		af->setPnt((const T *)&pnts[i], i);
	}
	m_features.push_back(af);
	return true;
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::clearFeatures()
{
	const int n = numFeatures();
	for(int i=0;i<n;++i) {
		delete m_features[i];
	}
	m_features.clear();
}

template<typename T, typename Tf>
int PCASimilarity<T, Tf>::featureDim() const
{
	if(numFeatures() < 1) {
		return 0;
	}
	
	return m_features[0]->numPnts();
}

template<typename T, typename Tf>
int PCASimilarity<T, Tf>::numFeatures() const
{ return m_features.size(); }

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::computeSimilarity()
{
	const int n = numFeatures();
	for(int i=0;i<n;++i) {
		m_features[i]->toLocalSpace();
	}
/// todo pca dimensionality reduction and k-mean clustering
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::getFeatureSpace(T * dst,
					const int & idx) const
{ m_features[idx]->getPCSpace(dst); }

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::getFeaturePoints(DenseMatrix<T> & dst,
					const int & idx,
					const int & dim) const
{ 
	if(dim==1) {
		const Tf * f = m_features[idx];
		const int np = f->numPnts();
		for(int i=0;i<np;++i) {
			f->getDataPoint(dst.column(i), i);
		}
	} else {
		const DenseMatrix<T> & src = m_features[idx]->dataPoints();
		dst.copy(src);
	}
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::getFeatureBound(T * dst,
					const int & idx,
					const int & dim) const
{ m_features[idx]->getBound(dst, dim); }

}

#endif
