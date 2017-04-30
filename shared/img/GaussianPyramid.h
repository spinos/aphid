/*
 *  GaussianPyramid.h
 *
 *  multi scale presentation of input signal
 *  start at level 0  
 *  apply low pass filter and scale down on each level
 *	reference 
 *  http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
 *  http://songho.ca/dsp/convolution/convolution.html
 *
 *  Created by jian zhang on 3/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_IMG_GAUSSIAN_PYRMAID_H
#define APH_IMG_GAUSSIAN_PYRMAID_H

#include <math/ATypes.h>
#include <img/BoxSampleProfile.h>
#include <img/ImageSpace.h>

namespace aphid {

namespace img {

template<typename T>
class GaussianPyramid {

/// 1:7
	int m_numLevels;
	
public:
	typedef Array3<T> SignalTyp;
	typedef Array2<T> SignalSliceTyp;
	
	GaussianPyramid();
	virtual ~GaussianPyramid();

	virtual void create(const SignalTyp & inputSignal);
	const float & aspectRatio() const;
	const int & numLevels() const;
	const SignalTyp & levelSignal(int level) const;
	Int2 levelSignalSize(int level) const;
	
/// find SampleFilterSize[lo] <= filterSize < SampleFilterSize[hi]	
/// mixing <- (filterSize - loFilterSize) / (hiFilterSize - loFilterSize)
/// filter size (1.0, 32.0), disable filtering if filter size == 0.0
	void getSampleProfile(BoxSampleProfile<T> * prof,
			const float & filterSize) const;
	
	T sample(BoxSampleProfile<T> * prof) const;
	
	const int & inputSignalNumCols() const;
	const int & inputSignalNumRows() const;
	
	void verbose() const;
	
protected:
	void applyBlurFilter(SignalTyp & dst, const SignalTyp & src);
	void blurVertical(SignalSliceTyp * dstSlice, const SignalSliceTyp & srcSlice,
							const int & m, const int & n);
	void blurHorizontal(SignalSliceTyp * dstSlice, const SignalSliceTyp & srcSlice,
							const int & m, const int & n);
	T blurredVertical(const SignalSliceTyp & slice,
					int iRow, int jCol) const;
	T blurredHorizontal(const SignalSliceTyp & slice,
					int iRow, int jCol) const;
					
	void scaleDown(SignalTyp & dst, const SignalTyp & src);
	T sampleStage(int level, const int & k,
				BoxSampleProfile<T> * prof) const;
/// without filter
	T sampleStage(int level, const int & k, 
				const float & coordU, const float & coordV) const;
	
/// channel[k] at level
/// dst has 2 ranks, 0 is u horizontal, 1 is v vertical
	void computeDerivative(SignalTyp & dst, int k, int level) const;
	void computeSobelDerivative(SignalTyp & dst, int k, int level) const;
/// channel[k]
	void getMinMax(T & vmin, T & vmax, int k) const;
	
private:
//// 0:5
	SignalTyp m_stage[7];
/// height (num rows) / width (num cols)
	float m_aspectRatio;
/// pixel size at each level
	static const float SampleFilterSize[7];
/// blur filter separated convolution
	static const T BlurKernel[5];
/// https://en.wikipedia.org/wiki/Image_derivatives
/// |-1 0 1|
/// |-2 0 2|
/// |-1 0 1|	
	static const T SobelUKernel[2][3];
/// |-1 -2 -1|
/// | 0  0  0|
/// | 1  2  1| 
	static const T SobelVKernel[2][3];
	
};

template<typename T>
const float GaussianPyramid<T>::SampleFilterSize[7] = {1.f, 2.f, 4.f, 8.f, 16.f, 32.f, 64.f};

/// (1 4 6.4 4 1) / 16.4
template<typename T>
const T GaussianPyramid<T>::BlurKernel[5] = {0.060976, 0.2439024, 0.3902439, 0.2439024, 0.060976};

template<typename T>
const T GaussianPyramid<T>::SobelUKernel[2][3] = {
{1.0, 2.0, 1.0}, {-1.0, 0.0, 1.0}
};

template<typename T>
const T GaussianPyramid<T>::SobelVKernel[2][3] = {
{-1.0, 0.0, 1.0}, {1.0, 2.0, 1.0}
};

template<typename T>
GaussianPyramid<T>::GaussianPyramid() :
m_numLevels(0)
{}

template<typename T>
GaussianPyramid<T>::~GaussianPyramid()
{}

template<typename T>
void GaussianPyramid<T>::create(const SignalTyp & inputSignal)
{
	m_aspectRatio = (float)inputSignal.numRows() / (float)inputSignal.numCols();
	
	m_stage[0] = inputSignal;
	m_numLevels = 1;
	
	SignalTyp cur;
	
	while(m_numLevels < 7
		&& m_stage[m_numLevels-1].numRows() > 8
		&& m_stage[m_numLevels-1].numCols() > 8 ) {
		
		applyBlurFilter(cur, m_stage[m_numLevels-1]);
		scaleDown(m_stage[m_numLevels], cur);
		
		m_numLevels++;
	}
}

template<typename T>
void GaussianPyramid<T>::applyBlurFilter(SignalTyp & dst, const SignalTyp & src)
{
	const int & m = src.numRows();
	const int & n = src.numCols();
	const int & p = src.numRanks();
	dst.create(m, n, p);
	
	SignalSliceTyp convol;
	convol.create(m, n);
	
	for(int k=0;k<p;++k) {
		const SignalSliceTyp * srcSlice = src.rank(k);
		SignalSliceTyp * dstSlice = dst.rank(k);
		
		blurVertical(&convol, *srcSlice, m, n);
		blurHorizontal(dstSlice, convol, m, n);

	}
}

template<typename T>
void GaussianPyramid<T>::blurVertical(SignalSliceTyp * dstSlice, const SignalSliceTyp & srcSlice,
							const int & m, const int & n)
{
	for(int j=0;j<n;++j) {
		T * colj = dstSlice->column(j);
		
		for(int i=0;i<m;++i) {
			colj[i] = blurredVertical(srcSlice, i, j);
		}
	}
}

template<typename T>
void GaussianPyramid<T>::blurHorizontal(SignalSliceTyp * dstSlice, const SignalSliceTyp & srcSlice,
							const int & m, const int & n)
{
	for(int j=0;j<n;++j) {
		T * colj = dstSlice->column(j);
		
		for(int i=0;i<m;++i) {
			colj[i] = blurredHorizontal(srcSlice, i, j);
		}
	}
}

template<typename T>
T GaussianPyramid<T>::blurredVertical(const SignalSliceTyp & slice,
					int iRow, int jCol) const
{
	const int limi = slice.numRows() - 1;
	T sum = 0;
	int ri;

	const T * colj = slice.column(jCol);
		
	for(int i=0;i<5;++i) {
		ri = iRow + i - 2;
		if(ri < 0) {
			ri = 0;
		}
		if(ri > limi) {
			ri = limi;
		}
		
		sum += colj[ri] * BlurKernel[i];
	}
	
	return sum;
	
}

template<typename T>
T GaussianPyramid<T>::blurredHorizontal(const SignalSliceTyp & slice,
					int iRow, int jCol) const
{
	const int limj = slice.numCols() - 1;
	T sum = 0;
	int rj;

	const T * colj = slice.column(jCol);
		
	for(int i=0;i<5;++i) {
		rj = jCol + i - 2;
		if(rj < 0) {
			rj = 0;
		}
		if(rj > limj) {
			rj = limj;
		}
		
		sum += slice.column(rj)[iRow] * BlurKernel[i];
	}
	
	return sum;
	
}
	
template<typename T>
void GaussianPyramid<T>::scaleDown(SignalTyp & dst, const SignalTyp & src)
{
	const int m = src.numRows() >> 1;
	const int n = src.numCols() >> 1;
	const int p = src.numRanks();
	dst.create(m, n, p);
	
	for(int k=0;k<p;++k) {
		const SignalSliceTyp * srcSlice = src.rank(k);
		SignalSliceTyp * dstSlice = dst.rank(k);
		
		for(int j=0;j<n;++j) {
			T * colj = dstSlice->column(j);
			
			for(int i=0;i<m;++i) {
				colj[i] = srcSlice->column(j<<1)[i<<1];
			}
		}
	}
}

template<typename T>
const float & GaussianPyramid<T>::aspectRatio() const
{ return m_aspectRatio; }

template<typename T>
const int & GaussianPyramid<T>::numLevels() const
{ return m_numLevels; }

template<typename T>
const Array3<T> & GaussianPyramid<T>::levelSignal(int level) const
{ return m_stage[level]; }

template<typename T>
Int2 GaussianPyramid<T>::levelSignalSize(int level) const
{ 
	const SignalTyp & lsig = levelSignal(level);
	return Int2(lsig.numCols(), lsig.numRows() ); 
}

template<typename T>
T GaussianPyramid<T>::sample(BoxSampleProfile<T> * prof) const
{
	if(prof->isTexcoordOutofRange() ) {
		return prof->_defaultValue;
	}
	
	if(prof->_loLevel < 0) {
		return sampleStage(0, prof->_channel, 
				prof->_uCoord, prof->_vCoord);
	}
	
	T loValue = sampleStage(prof->_loLevel, prof->_channel, prof);
	if(prof->_mixing < 1e-2f
		|| prof->_loLevel == prof->_hiLevel) {
		return loValue;
	}
	
	T hiValue = sampleStage(prof->_hiLevel, prof->_channel, prof);
	return loValue + (hiValue - loValue) * prof->_mixing;
}

template<typename T>
void GaussianPyramid<T>::getSampleProfile(BoxSampleProfile<T> * prof,
			const float & filterSize) const
{
	prof->_mixing = 0.f;
	int & lo = prof->_loLevel;
	int & hi = prof->_hiLevel;
	if(filterSize == 0.f) {
		lo = hi = -1;
		return;
	}
	
	if(filterSize <= 1.f) {
		lo = hi = 0;
		return;
	}
	
	lo = 0;
	hi = 1;
	while (hi < m_numLevels - 1) {
		if(filterSize >= SampleFilterSize[lo]
			&& filterSize < SampleFilterSize[hi]) {
			
			prof->_mixing = (filterSize - SampleFilterSize[lo]) / (SampleFilterSize[hi] - SampleFilterSize[lo]);
			return;
		}
		lo = hi;
		hi++;
	}
	lo = hi = m_numLevels - 1;
}

template<typename T>
T GaussianPyramid<T>::sampleStage(int level, const int & k, 
			const float & coordU, const float & coordV) const
{
	const SignalSliceTyp * slice = m_stage[level].rank(k);
	int u = coordU * slice->numCols();
	int v = coordV * slice->numRows();
	return slice->column(u)[v];
}

template<typename T>
T GaussianPyramid<T>::sampleStage(int level, const int & k,
				BoxSampleProfile<T> * prof) const
{
	const SignalSliceTyp * slice = m_stage[level].rank(k);
	const int & m = slice->numRows();
	const int & n = slice->numCols();
	float fu = prof->_uCoord * (float)n - .5f;
	float fv = prof->_vCoord * (float)m - .5f;
	int u0 = fu;
	int v0 = fv;
	int u1 = u0 + 1;
	int v1 = v0 + 1;
	
	if(u0 < 0) {
		u0 = 0;
	}
	if(v0 < 0) {
		v0 = 0;
	}
	if(u1 > n - 1) {
		u1 = n - 1;
	}
	if(v1 > m - 1) {
		v1 = m - 1;
	}
	
	fu -= u0;
	fv -= v0;
	if(fu < 0.f) {
		fu = 0.f;
	}
	if(fv < 0.f) {
		fv = 0.f;
	}
	
	T * box = prof->_box;
	
	box[0] = slice->column(u0)[v0];
	box[1] = slice->column(u0)[v1];
	box[2] = slice->column(u1)[v0];
	box[3] = slice->column(u1)[v1];
	
	box[0] += fv * (box[1] - box[0]);
	box[2] += fv * (box[3] - box[2]);
	
	return box[0] + fu * (box[2] - box[0]);
}

template<typename T>
void GaussianPyramid<T>::computeDerivative(SignalTyp & dst, int k, int level) const
{
	const int & m = levelSignal(level).numRows();
	const int & n = levelSignal(level).numCols();
	dst.create(m, n, 2);
	
	computeSobelDerivative(dst, k, level);

}

template<typename T>
void GaussianPyramid<T>::computeSobelDerivative(SignalTyp & dst, int k, int level) const
{
	const int & m = dst.numRows();
	const int & n = dst.numCols();
	
	const SignalSliceTyp * src = m_stage[level].rank(k);

	SignalSliceTyp * uDev = dst.rank(0);
	SignalSliceTyp * vDev = dst.rank(1);
	
	SignalSliceTyp convol;
	convol.create(m, n);
	
	convol.convoluteVertical(*src, SobelUKernel[0]);
	uDev->convoluteHorizontal(convol, SobelUKernel[1]);
	
	convol.convoluteVertical(*src, SobelVKernel[0]);
	vDev->convoluteHorizontal(convol, SobelVKernel[1]);
	
}

template<typename T>
const int & GaussianPyramid<T>::inputSignalNumCols() const
{ return m_stage[0].numCols(); }
	
template<typename T>
const int & GaussianPyramid<T>::inputSignalNumRows() const
{ return m_stage[0].numRows(); }

template<typename T>
void GaussianPyramid<T>::getMinMax(T & vmin, T & vmax, int k) const
{ 
	const SignalSliceTyp * slice = m_stage[m_numLevels - 1].rank(k);
	slice->getMinMax(vmin, vmax); 
}

template<typename T>
void GaussianPyramid<T>::verbose() const
{
	Int2 size0 = levelSignalSize(0);
	Int2 size1 = levelSignalSize(numLevels()-1);
	std::cout<<"\n GaussianPyramid n levels "<<numLevels()
		<<"\n size0 "<<size0.x<<" x "<<size0.y
		<<"\n size"<<numLevels()-1<<" "<<size1.x<<" x "<<size1.y
		<<"\n aspect ratio "<<aspectRatio()
		<<"\n n channels "<<levelSignal(0).numRanks();
		
}

}

}
#endif