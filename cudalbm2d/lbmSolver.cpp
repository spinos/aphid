#include <QtGui>

#include <math.h>

#include "lbmSolver.h"
#include "solver_implement.h"
#define LAT_W 192
#define LAT_H 120
#define LAT_LEN LAT_W * LAT_H
#define idx(x,y) ((y)*LAT_W+(x))
#define M_WALL 0
#define M_XOUT 1
#define M_YOUT 2
#define M_XIN 3
#define M_YIN 4
#define M_FLUID 5
#define M_SETU 6

const float visc = 0.009f;

RenderThread::RenderThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
	
	wall = new unsigned char[LAT_LEN];
	pixel = new uchar[LAT_LEN*3];
	impulse = new float[LAT_LEN * 2];
		
	int x, y, i;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {
			i = idx(x,y);
			wall[i] = M_FLUID;
			impulse[i*2] = impulse[i*2 + 1] = 0.f;
		}
	}

	for (x = 0; x < LAT_W; x++) {
		wall[idx(x,0)] = wall[idx(x,LAT_H-1)] = M_WALL;
	}

	for (y = 0; y < LAT_H; y++) {
		wall[idx(0,y)] = wall[idx(LAT_W-1,y)] = M_WALL;
	}
	
	for (y = 1; y < LAT_H-1; y++) {
		wall[idx(0,y)] = M_WALL;
		wall[idx(LAT_W-1,y)] = M_WALL;
	}
	
	initializeSolverData(LAT_W, LAT_H);
}

RenderThread::~RenderThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
    
}

int RenderThread::solverWidth() const
{
	return LAT_W;
}

int RenderThread::solverHeight() const
{
	return LAT_H;
}


void RenderThread::render()
{
    QMutexLocker locker(&mutex);

	this->resultSize = QSize(LAT_W, LAT_H);
	

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void RenderThread::run()
{
    forever {
        
	
	simulate();
	simulate();
	simulate();
	simulate();
      
        if (abort) {
            destroySolverData();
            return;
        }
        getDisplayField(LAT_W, LAT_H, wall, pixel);

        QImage image(pixel, LAT_W, LAT_H, QImage::Format_RGB888);
		
		if (!restart)
				emit renderedImage(image, 0);

		mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
    }
}

void RenderThread::simulate()
{
	advanceSolver(LAT_W, LAT_H, impulse, wall);
}

void RenderThread::addImpulse(int x, int y, float vx, float vy)
{
	int gi = idx(x,y);
	impulse[gi * 2] = vx;
	impulse[gi * 2 + 1] = vy;	
}

void RenderThread::addObstacle(int x, int y)
{
	int gi = idx(x,y);
	wall[gi] = M_WALL;
}
