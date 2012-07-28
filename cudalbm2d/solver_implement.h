#ifndef SOLVER_IMPLEMENT_H
#define SOLVER_IMPLEMENT_H


extern "C" void showScalarField(int width, int height, float*pImage, unsigned char*outImage);
extern "C" void advectScalarField(int width, int height, float*u, float*field);

#endif        //  #ifndef SOLVER_IMPLEMENT_H

