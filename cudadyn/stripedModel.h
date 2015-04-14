#ifndef STRIPEDMODEL_H
#define STRIPEDMODEL_H
#include <AllMath.h>
inline unsigned extractElementInd(unsigned combined)
{ return ((combined<<7)>>7); }

inline unsigned extractObjectInd(unsigned combined)
{ return (combined>>24); }

inline Vector3F tetrahedronCenter(Vector3F * p, unsigned * v, unsigned * pntOffset, unsigned * indOffset, unsigned i)
{
	unsigned objectI = extractObjectInd(i);
	unsigned elementI = extractElementInd(i);
	
	unsigned pStart = pntOffset[objectI];
	unsigned iStart = indOffset[objectI];
	
	Vector3F r = p[pStart + v[iStart + elementI * 4]];
    r += p[pStart + v[iStart + elementI * 4 + 1]];
    r += p[pStart + v[iStart + elementI * 4 + 2]];
    r += p[pStart + v[iStart + elementI * 4 + 3]];
    r *= .25f;
    
	return r;
}
#endif        //  #ifndef STRIPEDMODEL_H

