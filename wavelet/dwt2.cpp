/*
 *  dwt2.cpp
 *  
 *
 *  Created by jian zhang on 9/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "dwt2.h"
#include <AllMath.h>

namespace aphid {

namespace wla {

void circleShift1(float * x, const int & n, const int & p)
{
	if(p==0) 
		return;
		
	VectorN<float> b;
	b.create(x, n);
	b.circshift(p);
	memcpy(x, b.v(), 4 * n );
	
}

void circleShift2(Array2<float> * x, const int & p)
{
	const int & m = x->numRows();
	const int & n = x->numCols();
	
	for(int i=0; i<n; ++i) {
		circleShift1(x->column(i), m, p);
	}
}

void afbRow(const Array2<float> * x, Array2<float> * lo, Array2<float> * hi)
{
	const int & nrow = x->numRows();
	const int & ncol = x->numCols();
	const int nsubrow = nrow / 2;
	
	lo->create(nsubrow, ncol);
	hi->create(nsubrow, ncol);
	
	VectorN<float> bcol;
	
	float * locol;
	float * hicol;
	int m;
	for(int i=0; i<ncol;++i) {
		bcol.copy(x->column(i), nrow, -5);
	
		afb(bcol.v(), nrow, locol, hicol, m);
		
		lo->copyColumn(i, locol);
		hi->copyColumn(i, hicol);
		
		delete[] locol;
		delete[] hicol;
	}
}

void afbRowflt(const Array2<float> * x,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * hi)
{
	const int & nrow = x->numRows();
	const int & ncol = x->numCols();
	const int nsubrow = nrow / 2;
	
	lo->create(nsubrow, ncol);
	hi->create(nsubrow, ncol);
	
	VectorN<float> bcol;
	VectorN<float> locol;
	VectorN<float> hicol;

	for(int i=0; i<ncol;++i) {
		bcol.copy(x->column(i), nrow, -5);
	
		afbflt(bcol.v(), nrow, locol, hicol, flt);
		
		lo->copyColumn(i, locol.v() );
		hi->copyColumn(i, hicol.v() );
		
	}
}

void sfbRow(Array2<float> * y, Array2<float> * lo, Array2<float> * hi)
{
	const int & nrow = lo->numRows();
	const int & ncol = lo->numCols();
	y->create(nrow*2, ncol);
	
	float * ycol;
	int n;
	for(int i=0; i<ncol;++i) {
		sfb(lo->column(i), hi->column(i), nrow,
			ycol, n);
		
		y->copyColumn(i, ycol);
		
		delete[] ycol;
	}
}

void sfbRowflt(Array2<float> * y, 
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * hi)
{
	const int & nrow = lo->numRows();
	const int & ncol = lo->numCols();
	y->create(nrow*2, ncol);
	
	VectorN<float> ycol;
	VectorN<float> locol;
	VectorN<float> hicol;
	
	for(int i=0; i<ncol;++i) {
	
		locol.copy(lo->column(i), nrow);
		hicol.copy(hi->column(i), nrow);
		
		sfbflt(ycol,
			locol, hicol,
			flt);
		
		y->copyColumn(i, ycol.v() );
		
	}
}

void afb2(const Array2<float> * x,
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi)
{
	Array2<float> L;
	Array2<float> H;
/// filter along columns of x
	afbRow(x, &L, &H);
	
/// transpose to filter along columns
	L.transpose();
	H.transpose();
	
	afbRow(&L, lo, lohi);
	afbRow(&H, hilo, hihi);
	
/// transpose back
	lo->transpose();
	lohi->transpose();
	hilo->transpose();
	hihi->transpose();
}

void sfb2(Array2<float> * y,
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi)
{
	Array2<float> L;
	Array2<float> H;
	
/// transpose to filter along columns
	lo->transpose();
	lohi->transpose();
	hilo->transpose();
	hihi->transpose();
	
	sfbRow(&L, lo, lohi);
	sfbRow(&H, hilo, hihi);
	
/// transpose back to along columns
	L.transpose();
	H.transpose();
	
	sfbRow(y, &L, &H);
}

void afb2flt(const Array2<float> * x,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi)
{
	Array2<float> L;
	Array2<float> H;
/// filter along columns of x
	afbRowflt(x, flt, &L, &H);
	
/// transpose to filter along columns
	L.transpose();
	H.transpose();
	
	afbRowflt(&L, flt, lo, lohi);
	afbRowflt(&H, flt, hilo, hihi);
	
/// transpose back
	lo->transpose();
	lohi->transpose();
	hilo->transpose();
	hihi->transpose();
}

void sfb2flt(Array2<float> * y,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi)
{
	Array2<float> L;
	Array2<float> H;
	
/// transpose to filter along columns
	lo->transpose();
	lohi->transpose();
	hilo->transpose();
	hihi->transpose();
	
	sfbRowflt(&L, flt, lo, lohi);
	sfbRowflt(&H, flt, hilo, hihi);
	
/// transpose back to along columns
	L.transpose();
	H.transpose();
	
	sfbRowflt(y, flt, &L, &H);
}

}

}