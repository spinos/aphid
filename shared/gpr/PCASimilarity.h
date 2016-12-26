/*
 *  PCASimilarity.h
 *  
 *  similarity search via pca
 *  separate via kmean-clustering
 *
 *  Created by jian zhang on 12/20/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PCA_SIMILARITY_H
#define APH_PCA_SIMILARITY_H

#include <gpr/PCAReduction.h>
#include <math/kmean.h>
#include <math/deviate_mean.h>

namespace aphid {

template<typename T, typename Tf>
class PCASimilarity {

	KMeansClustering2<T> m_cluster;
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
	
	bool separateFeatures(int nsep=2);
	
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
	
	const int * groupIndices() const;
	
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
bool PCASimilarity<T, Tf>::separateFeatures(int nsep)
{
	const int n = numFeatures();
	for(int i=0;i<n;++i) {
		m_features[i]->toLocalSpace();
	}

	const int fd = featureDim() * Tf::numVars();
	PCAReduction<float> dimred;
	dimred.createX(fd, n);
	for(int i=0;i<n;++i) {
		dimred.setXi(m_features[i]->dataPoints(), i);
	}
	
	DenseMatrix<float> redX;
	dimred.compute(redX);
	
	std::cout<<" reduced x "<<redX;
	
	const T dev = deviate_from_mean(redX, 2);
	int K = nsep;
	if(dev < 1.0e-2) {
		std::cout<<" PCASimilarity found not enough deviation "<<dev
					<<" to separate features, return in 1 group ";		
		K = 1;
	}
	
	if(nsep>n) {
		std::cout<<" PCASimilarity has not enough features to separate into "<<nsep
					<<" groups, return in "<<n<<" groups ";
		K = n;
	}
	
	m_cluster.setKND(K, n, 2);
	if(!m_cluster.compute(redX) ) {
		std::cout<<"\n PCASimilarity keam failed ";
		return false;
	}
	return true;
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
	const Tf * f = m_features[idx];
	const int np = f->numPnts();
		
	if(dim==1) {
		for(int i=0;i<np;++i) {
			f->getScaledDataPoint(dst.column(i), i);
		}
		
	} else {
		DenseVector<T> apnt(Tf::numVars() );
		for(int i=0;i<np;++i) {
			f->getScaledDataPoint(apnt.raw(), i);
			dst.copyRow(i, apnt.v() );
			
		}
		
	}
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::getFeatureBound(T * dst,
					const int & idx,
					const int & dim) const
{ m_features[idx]->getBound(dst, dim); }

template<typename T, typename Tf>
const int * PCASimilarity<T, Tf>::groupIndices() const
{ return m_cluster.groupIndices(); }

}

#endif
