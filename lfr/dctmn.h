#ifndef DCTMN_H
#define DCTMN_H

#include <deque>
#include "LfMachine.h"
#include "regr.h"
#include "psnr.h"
#include "dct.h"
#include <boost/thread.hpp>
/// f2c macros conflict
#define _WIN32
#include <ExrImage.h> 

namespace lfr {

template<int NumThread, typename T>
class DictionaryMachine : public LfMachine {

/// signal
	DenseVector<T> * m_y;
/// coefficients
	DenseVector<T> * m_beta;
/// sparse indices
	DenseVector<int> * m_ind;
	DenseVector<T> * m_ui;
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
/// previous epoch
    DenseMatrix<T> * m_pre0A;
    DenseMatrix<T> * m_pre0B;
/// least angle regression
	LAR<T> * m_lar;
    
	T m_sqePt[NumThread];
	SquareErr<T> m_sqeWorker[NumThread];
	LAR<T> * m_larPt[NumThread];
	DenseVector<T> * m_yPt[NumThread];
	DenseVector<T> * m_betaPt[NumThread];
	DenseVector<int> * m_indPt[NumThread];
	DenseMatrix<T> * m_batchAPt[NumThread];
    DenseMatrix<T> * m_batchBPt[NumThread];
	
	int m_atomSize;
	int m_pret;
    int m_epoch;
    T m_lambda;
    T m_totalErr;
    
public:
    DictionaryMachine(LfParameter * param);
    virtual ~DictionaryMachine();
    
    virtual void initDictionary();
    virtual void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	virtual void fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor);
	virtual void preLearn();
    virtual void learn(const aphid::ExrImage * image, int ibegin, int iend);
    virtual void updateDictionary(const aphid::ExrImage * image, int t);
    virtual void cleanDictionary();
/// keep coefficients of current epoch
/// at end of epoch, set A and B to current epoch, 
/// zero current epoch 
    virtual void recycleData(); 
    virtual T computePSNR(const aphid::ExrImage * image, int iImage);

protected:
    
private:
    void learnPt(const int iThread, const aphid::ExrImage * image, const int workBegin, const int workEnd);
    void computeErrPt(const int iThread, const aphid::ExrImage * image, const int workBegin, const int workEnd);
    void fillPatchPt(const int iThread, unsigned * line, const int workBegin, const int workEnd,
						const int dimx, const int s, const int imageW);
	void computeLambda(int t);
};

template<int NumThread, typename T>
DictionaryMachine<NumThread, T>::DictionaryMachine(LfParameter * param) :
    LfMachine(param)
{
    const int m = param->dimensionOfX();
	const int p = param->dictionaryLength();
    m_D = new DenseMatrix<T>(m, p);
    m_G = new DenseMatrix<T>(p, p);
	m_A = new DenseMatrix<T>(p, p);
	m_B = new DenseMatrix<T>(m, p);
	m_lar = new LAR<T>(m_D, m_G);
	m_y = new DenseVector<T>(m);
	m_beta = new DenseVector<T>(p);
	m_ind = new DenseVector<int>(p);
	m_batchA = new DenseMatrix<T>(p, p);
	m_batchB = new DenseMatrix<T>(m, p);
	m_pre0A = new DenseMatrix<T>(p, p);
	m_pre0B = new DenseMatrix<T>(m, p);
    m_ui = new DenseVector<T>(m);
    m_epoch = -1;
    int i=0;
    for(;i<NumThread;++i) {
        m_sqeWorker[i].create(m_D);
        m_larPt[i] = new LAR<T>(m_D, m_G);
        m_yPt[i] = new DenseVector<T>(m);
        m_betaPt[i] = new DenseVector<T>(p);
        m_indPt[i] = new DenseVector<int>(p);
        m_batchAPt[i] = new DenseMatrix<T>(p, p);
        m_batchBPt[i] = new DenseMatrix<T>(m, p);
    }
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
    delete m_batchA;
    delete m_batchB;
    delete m_pre0A;
    delete m_pre0B;
    delete m_ui;
    int i=0;
    for(;i<NumThread;++i) {
        delete m_larPt[i];
        delete m_yPt[i];
        delete m_betaPt[i];
        delete m_indPt[i];
        delete m_batchAPt[i];
        delete m_batchBPt[i];
    }
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::initDictionary()
{
    m_atomSize = param()->atomSize();
	const int k = m_D->numColumns();
	const int m = m_D->numRows();
	int i, j, p, q;
	for(i=0;i<k;i++) {
/// init D with random signal 
        float * d = m_D->column(i);
        aphid::ExrImage * img = param1()->openImage(param()->randomImageInd());
        
        img->getTile(d, rand(), m_atomSize);
        p = rand() & 7;
        q = rand() & 7;
        Dct<T>::EnhanceBasisFunc(d,         m_atomSize, p, q, 0.5);
        p = rand() & 7;
        q = rand() & 7;
        Dct<T>::EnhanceBasisFunc(&d[m/3],   m_atomSize, p, q, 0.5);
        p = rand() & 7;
        q = rand() & 7;
        Dct<T>::EnhanceBasisFunc(&d[m/3*2], m_atomSize, p, q, 0.5);
        
	}
	
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary();
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::preLearn()
{
	m_A->setZero();
	m_A->addDiagonal(1e-5);
	m_B->copy(*m_D);
/// B0 <- t0 * D
	m_B->scale(1e-5);
	
	m_batchA->setZero();
	m_batchB->setZero();
    m_pre0A->setZero();
    m_pre0B->setZero();
    m_pret = 0;
    m_epoch = -1;
    m_lambda = 1e-5;
    int i=0;
    for(;i<NumThread;++i) {
        m_batchAPt[i]->setZero();
        m_batchBPt[i]->setZero();
    }
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::learnPt(const int iThread, const aphid::ExrImage * image, const int workBegin, const int workEnd)
{
    const int k = m_D->numColumns();
    int i = workBegin;
    for(;i<=workEnd;++i) {
        image->getTile(m_yPt[iThread]->raw(), i, m_atomSize);

        m_larPt[iThread]->lars(*m_yPt[iThread], *m_betaPt[iThread], *m_indPt[iThread], m_lambda);
	
        int nnz = 0;
        int i=0;
        for(;i<k;++i) {
            if((*m_indPt[iThread])[i] < 0) break;
            nnz++;
        }
        if(nnz < 1) continue;
        
        sort<T, int>(m_indPt[iThread]->raw(), m_betaPt[iThread]->raw(), 0, nnz-1);
        
/// A <- A + beta * beta^t
	    m_batchAPt[iThread]->rank1Update(*m_betaPt[iThread], *m_indPt[iThread], nnz);
/// B <- B + y * beta^t
	    m_batchBPt[iThread]->rank1Update(*m_yPt[iThread], *m_betaPt[iThread], *m_indPt[iThread], nnz);
	}
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::learn(const aphid::ExrImage * image, int ibegin, int iend)
{
#if 1
    const int workSize = (iend - ibegin + 1) / NumThread;
    boost::thread learnThread[NumThread];
    int workBegin, workEnd, i=0;
    for(;i<NumThread;++i) {
        workBegin = ibegin + i * workSize;
        workEnd = (i== NumThread - 1) ? iend: workBegin + workSize - 1;
        learnThread[i] = boost::thread( boost::bind(&DictionaryMachine<NumThread, T>::learnPt, 
            this, i, image, workBegin, workEnd) );
    }
    
    for(i=0;i<NumThread;++i)
		learnThread[i].join();
	
#else
    const int k = m_D->numColumns();
    int j = ibegin;
    for(;j<=iend;++j) {
        image->getTile(m_y->raw(), j, m_atomSize);

        m_lar->lars(*m_y, *m_beta, *m_ind, m_lambda);
	
        int nnz = 0;
        int i=0;
        for(;i<k;++i) {
            if((*m_ind)[i] < 0) break;
            nnz++;
        }
        if(nnz < 1) continue;
        
        sort<T, int>(m_ind->raw(), m_beta->raw(), 0, nnz-1);
        
/// A <- A + beta * beta^t
	    m_batchA->rank1Update(*m_beta, *m_ind, nnz);
/// B <- B + y * beta^t
	    m_batchB->rank1Update(*m_y, *m_beta, *m_ind, nnz);
	}
#endif
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::updateDictionary(const aphid::ExrImage * image, int t)
{
    T sc = 1.0;
    //if(t>0) 
        m_pret = t;
/// scaling A and B from previous batch
    int tn = t+1;
    if(tn < 256) tn = tn * 256;
    else tn = 256 * 256 + t - 256;
        sc = T(tn+1-256)/T(tn+1);
        
/// accumulate pre-thread works
    int i = 0;
#if 1
    for(;i<NumThread;++i) {
        m_batchA->add(*m_batchAPt[i]);
        m_batchB->add(*m_batchBPt[i]);
        m_batchAPt[i]->setZero();
        m_batchBPt[i]->setZero();
    }
#endif
    
    if(t>1) {
/// scale the past
        //m_A->scale(sc);
        //m_B->scale(sc);
/// slow down early steps
        m_batchA->scale(0.01);
        m_batchB->scale(0.01);
    }
    
/// add average of batch
	m_A->add(*m_batchA, 1.0/256.0);
    m_B->add(*m_batchB, 1.0/256.0);
    m_pre0A->add(*m_batchA, 1.0/256.0);
    m_pre0B->add(*m_batchB, 1.0/256.0);
    m_batchA->setZero();
    m_batchB->setZero();
    
	const int p = m_D->numColumns();
	
    for (i = 0; i<p; ++i) {
        const T Aii = m_A->column(i)[i];
        if (Aii > 1e-8) {
            DenseVector<T> di(m_D->column(i), m_D->numRows());
            DenseVector<T> ai(m_A->column(i), m_A->numRows());
            DenseVector<T> bi(m_B->column(i), m_B->numRows());
/// ui <- (bi - D * ai) / Aii + di
            m_D->mult(*m_ui, ai, -1.0);
            m_ui->add(bi);
            m_ui->scale(1.0/Aii);
            m_ui->add(di);
/// di <- ui / max(|| ui ||, 1)	?			
            float unm = m_ui->norm();
            if(unm > 1.0) 
            	m_ui->scale(1.f/unm);
            
            m_D->copyColumn(i, m_ui->v());
       }
    }
    m_D->AtA(*m_G);	
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::cleanDictionary()
{
	const int k = m_D->numColumns();
	const int m = m_D->numRows();
    const int s = param()->atomSize();
	int i, j, l, p, q;
    
    int ncld = 0;
	for (i = 0; i<k; ++i) {
/// lower part of G
		for (j = i; j<k; ++j) {
			bool toClean = false;
			if(j==i) {
/// diagonal part
				toClean = absoluteValue<T>( m_G->column(i)[j] ) < 1e-6;
			}
			else {
				float ab = m_G->column(i)[i] * m_G->column(j)[j];
				toClean = ( absoluteValue<T>( m_G->column(i)[j] ) / sqrt( ab ) ) > 0.9999;
			}
			if(toClean) {
			    ncld++;
/// D_j <- randomly choose signal element
				DenseVector<T> dj(m_D->column(j), m_D->numRows() );
				
				//aphid::ExrImage * img = param->openImage(param->randomImageInd());
                //img->getTile(dj.raw(), rand(), s);
                p = rand() & 3;
                q = rand() & 3;
                Dct<T>::AddBasisFunc(dj.raw(),         s, p, q, 0.5);
                p = rand() & 3;
                q = rand() & 3;
                Dct<T>::AddBasisFunc(&dj.raw()[m/3],   s, p, q, 0.5);
                p = rand() & 3;
                q = rand() & 3;
                Dct<T>::AddBasisFunc(&dj.raw()[m/3*2], s, p, q, 0.5);
                
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
	if(ncld>0) {
	    std::cout<<"\n n cleaned atoms "<<ncld;
	    m_G->addDiagonal(1e-10);
	}
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::fillPatchPt(const int iThread, unsigned * line, 
												const int workBegin, const int workEnd,
												const int dimx, const int s, const int imageW)
{
	const int k = param()->dictionaryLength();
	int i, j;
	for(j= workBegin;j<= workEnd;j++) {
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
void DictionaryMachine<NumThread, T>::dictionaryAsImage(unsigned * imageBits, int imageW, int imageH)
{
    const int s = param()->atomSize();
	const int dimx = imageW / s;
	const int dimy = imageH / s;
	unsigned * line = imageBits;
	
#if 1
	const int nt = (dimy < NumThread) ? dimy : NumThread;
	const int workSize = dimy / nt;
	boost::thread fillThread[NumThread];
	int workBegin, workEnd, tid = 0;
	for(;tid<nt;++tid) {
        workBegin = tid * workSize;
        workEnd = (tid== nt - 1) ? dimy - 1: workBegin + workSize - 1;
        fillThread[tid] = boost::thread( boost::bind(&DictionaryMachine<NumThread, T>::fillPatchPt, 
            this, tid, &line[workBegin * imageW * s], workBegin, workEnd, dimx, s, imageW) );
    }
	
	for(tid=0;tid<nt;++tid)
		fillThread[tid].join();
		
#else
	int i, j;
	const int k = param()->dictionaryLength();
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
#endif
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor)
{
	DenseVector<unsigned> scanline(&imageBits[iLine * imageW], imageW);
	scanline.setZero();
	const int k = m_D->numColumns();
	const float ratio = (float)k / imageW;
			
	int i = 0;
	for(;i<imageW;++i) {
		if((*m_ind)[i*ratio] < 0) break;
		scanline[i] = fillColor;
	}
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::computeErrPt(const int iThread, const aphid::ExrImage * image, const int workBegin, const int workEnd)
{
    T sum = 0;
    int i = workBegin;
    for(;i<=workEnd;++i) {
        image->getTile(m_yPt[iThread]->raw(), i, m_atomSize);
        m_larPt[iThread]->lars(*m_yPt[iThread], *m_betaPt[iThread], *m_indPt[iThread], m_lambda);
        sum += m_sqeWorker[iThread].compute(*m_yPt[iThread], *m_betaPt[iThread], *m_indPt[iThread]);
    }
    m_sqePt[iThread] = sum;
}

template<int NumThread, typename T>
T DictionaryMachine<NumThread, T>::computePSNR(const aphid::ExrImage * image, int iImage)
{
    const int numPatches = param()->imageNumPatches(iImage);
    const int workSize = numPatches / NumThread;
    boost::thread errThread[NumThread];
    int workBegin, workEnd, i=0;
    for(;i<NumThread;++i) {
        workBegin = i * workSize;
        workEnd = (i== NumThread - 1) ? numPatches - 1: workBegin + workSize - 1;
        errThread[i] = boost::thread( boost::bind(&DictionaryMachine<NumThread, T>::computeErrPt, 
            this, i, image, workBegin, workEnd) );
    }
    
    for(i=0;i<NumThread;++i)
		errThread[i].join();
	
	T sum = 0.0;
	for(i=0;i<NumThread;++i)
		sum += m_sqePt[i];
	
    return 10.0 * log10( T(1.0) / (1e-10 + sum / param()->totalNumPixels() ) );
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::recycleData()
{
    m_epoch++;
    
    m_A->copy(*m_pre0A);
    m_B->copy(*m_pre0B);
    
    //if(m_epoch>0) 
    {
        m_A->scale(0.7);
        m_B->scale(0.7);
    }
    
    m_pre0A->setZero();
	m_pre0B->setZero();
	
}

template<int NumThread, typename T>
void DictionaryMachine<NumThread, T>::computeLambda(int t)
{
    if(t<10) m_lambda = 0.0;
    else {
        T tt = T(t-9) / 1000.0;
        if(tt > 1.0) tt = 1.0;
        m_lambda = 0.003 * tt * tt;
        std::cout<<" lambda "<<m_lambda;
    }
}

}
#endif        //  #ifndef DCTMN_H

