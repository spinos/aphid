#ifndef DCTMN_H
#define DCTMN_H

#include "LfParameter.h"
#include "regr.h"
#include "psnr.h"
/// f2c macros conflict
#define _WIN32
#include <ExrImage.h>

namespace lfr {

template<int NumThread, typename T>
class DictionaryMachine {

/// signal
	DenseVector<T> * m_y;
/// coefficients
	DenseVector<T> * m_beta;
/// sparse indices
	DenseVector<int> * m_ind;
/// dictionary
	DenseMatrix<T> * m_D;
/// gram of D
	DenseMatrix<T> * m_G;
/// beta * beta^t
	DenseMatrix<T> * m_A;
/// X * beta^t
	DenseMatrix<T> * m_B;
/// per-batch A and B
    DenseMatrix<T> * m_batchA;
    DenseMatrix<T> * m_batchB;
/// least angle regression
	LAR<T> * m_lar;
/// peak signal-to-noise ratio
	Psnr<T> * m_errorCalc;
    
	int m_atomSize;
public:
    DictionaryMachine(const int m, const int p);
    virtual ~DictionaryMachine();
    
    void initDictionary(LfParameter * param);
    void cleanDictionary(LfParameter * param);
    void preLearn();
    void learn(const ExrImage * image, int ibegin, int iend);
    void updateDictionary(bool forceClean);
    
    void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH, 
                            const LfParameter * param);
	
protected:
    
private:
    void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
	
};

template<int NumThread, typename T>
DictionaryMachine<NumThread, T>::DictionaryMachine(const int m, const int p)
{
    m_D = new DenseMatrix<T>(m, p);
	m_G = new DenseMatrix<T>(p, p);
	m_A = new DenseMatrix<T>(p, p);
	m_B = new DenseMatrix<T>(m, p);
	m_lar = new LAR<T>(m_D, m_G);
	m_y = new DenseVector<T>(m);
	m_beta = new DenseVector<T>(p);
	m_ind = new DenseVector<int>(p);
	m_errorCalc = new Psnr<T>(m_D);
	m_batchA = new DenseMatrix<T>(p, p);
	m_batchB = new DenseMatrix<T>(m, p);
}

template<int NumThread, typename T>
DictionaryMachine<NumThread, T>::~DictionaryMachine()
{
    delete m_D;
    delete m_G;
    delete m_A;
    delete m_B;
    delete m_lar;
    delete m_y;
    delete m_beta;
    delete m_ind;
    delete m_errorCalc;
    delete m_batchA;
    delete m_batchB;
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::initDictionary(LfParameter * param)
{
    m_atomSize = param->atomSize();
	const int k = m_D->numColumns();
	
	int i, j;
	for(i=0;i<k;i++) {
/// init D with random signal 
        float * d = m_D->column(i);
        ExrImage * img = param->openImage(param->randomImageInd());
        
        img->getTile(d, rand(), m_atomSize);
	}
	
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary(param);
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::preLearn()
{
	m_A->setZero();
	m_A->addDiagonal(1e-5);
	m_B->copy(*m_D);
	m_B->scale(1e-7);
	
	m_batchA->setZero();
	m_batchB->setZero();
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::learn(const ExrImage * image, int ibegin, int iend)
{
    const int k = m_D->numColumns();
    int j = ibegin;
    for(;j<=iend;++j) {
        image->getTile(m_y->raw(), j, m_atomSize);

        m_lar->lars(*m_y, *m_beta, *m_ind, 0.0);
	
        int nnz = 0;
        int i=0;
        for(;i<k;++i) {
            if((*m_ind)[i] < 0) break;
            nnz++;
        }
        if(nnz < 1) break;
        
        sort<T, int>(m_ind->raw(), m_beta->raw(), 0, nnz-1);
        
/// A <- A + beta * beta^t
	    m_batchA->rank1Update(*m_beta, *m_ind, nnz);
/// B <- B + y * beta^t
	    m_batchB->rank1Update(*m_y, *m_beta, *m_ind, nnz);
	}
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::updateDictionary(bool forceClean)
{
/// combine a batch weighted to smaller step
    m_batchA->scale(1.0/500);
    m_batchB->scale(1.0/500);
/// reduce A increases chance to clean an atom
/// blindly select a patch can be equally bad, lower than 0.9 is too random
/// A accumulated after each iteration, lesser chance to clean
    float sc = 0.93;
    //float sc = float(niter+10000 - 256)/float(niter+10000);
    m_A->scale(sc);
    m_B->scale(sc);
    m_A->add(*m_batchA);
    m_B->add(*m_batchB);
    m_batchA->setZero();
    m_batchB->setZero();
    
	const int p = m_D->numColumns();
	DenseVector<float> ui(m_D->numRows());
	int i;
		for (i = 0; i<p; ++i) {
			const float Aii = m_A->column(i)[i];
			if (Aii > 1e-6) {
				DenseVector<float> di(m_D->column(i), m_D->numRows());
				DenseVector<float> ai(m_A->column(i), m_A->numRows());
				DenseVector<float> bi(m_B->column(i), m_B->numRows());
/// ui <- (bi - D * ai) / Aii + di
				m_D->mult(ui, ai, -1.0f);
				ui.add(bi);
				ui.scale(1.0f/Aii);
				ui.add(di);
/// di <- ui / max(|| ui ||, 1)				
				float unm = ui.norm();
				if(unm > 1.0) 
					ui.scale(1.f/unm);
				
				m_D->copyColumn(i, ui.v());
		   }
		   else {
		       if(forceClean) {
		           DenseVector<float> di(m_D->column(i), m_D->numRows());
		           di.setZero();
		       }
		   }
		}		
	
	m_D->normalize();
	m_D->AtA(*m_G);
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::cleanDictionary(LfParameter * param)
{
    const int k = m_D->numColumns();
    const int s = param->atomSize();
	int i, j, l;
	for (i = 0; i<k; ++i) {
/// lower part of G
		for (j = i; j<k; ++j) {
			bool toClean = false;
			if(j==i) {
/// diagonal part
				toClean = absoluteValue<T>( m_G->column(i)[j] ) < 1e-7;
			}
			else {
				float ab = m_G->column(i)[i] * m_G->column(j)[j];
				toClean = ( absoluteValue<T>( m_G->column(i)[j] ) / sqrt( ab ) ) > 0.9999;
			}
			if(toClean) {
/// D_j <- randomly choose signal element
				DenseVector<T> dj(m_D->column(j), m_D->numRows() );
				
				ExrImage * img = param->openImage(param->randomImageInd());
                img->getTile(dj.raw(), rand(), s);
                
				dj.normalize();
/// G_j <- D^t * D_j
				DenseVector<T> gj(m_G->column(j), k);
				m_D->multTrans(gj, dj);
/// copy to diagonal line of G
				for (l = 0; l<k; ++l)
					m_G->column(l)[j] = m_G->column(j)[l];
			}
		}
	}
	m_G->addDiagonal(1e-8);
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::dictionaryAsImage(unsigned * imageBits, int imageW, int imageH, 
                                        const LfParameter * param)
{
    const int s = param->atomSize();
	const int dimx = imageW / s;
	const int dimy = imageH / s;
	const int k = param->dictionaryLength();
	int i, j;
	unsigned * line = imageBits;
	for(j=0;j<dimy;j++) {
		for(i=0;i<dimx;i++) {
			const int ind = dimx * j + i;
			if(ind < k) {
			    float * d = m_D->column(ind);
			    fillPatch(&line[i * s], d, s, imageW);
			}
		}
		line += imageW * s;
	}
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::fillPatch(unsigned * dst, float * color, int s, int imageW, int rank)
{
    const int stride = s * s;
	int crgb[3];
	int i, j, k;
	unsigned * line = dst;
	for(j=0;j<s; j++) {
		for(i=0; i<s; i++) {
			unsigned v = 255<<24;
			for(k=0;k<rank;k++) {				
				crgb[k] = 8 + 500 * color[(j * s + i) + k * stride];
				crgb[k] = std::min<int>(crgb[k], 255);
				crgb[k] = std::max<int>(crgb[k], 0);
			}
			v = v | ( crgb[0] << 16 );
			v = v | ( crgb[1] << 8 );
			v = v | ( crgb[2] );
			line[i] = v;
		}
		line += imageW;
	}
}

}
#endif        //  #ifndef DCTMN_H

