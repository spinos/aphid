#ifndef STRIPEDMODEL_H
#define STRIPEDMODEL_H

inline unsigned extractElementInd(unsigned combined)
{ return ((combined<<7)>>7); }

inline unsigned extractObjectInd(unsigned combined)
{ return (combined>>24); }
#endif        //  #ifndef STRIPEDMODEL_H

