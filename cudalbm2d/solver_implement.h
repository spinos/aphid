#ifndef SOLVER_IMPLEMENT_H
#define SOLVER_IMPLEMENT_H

extern "C" void initializeSolverData(int width, int height);
extern "C" void destroySolverData();
extern "C" void getDisplayField(int width, int height, unsigned char * obstable, unsigned char*outImage);
extern "C" void advanceSolver(int width, int height, float *impulse, unsigned char * obstable);
#endif        //  #ifndef SOLVER_IMPLEMENT_H

