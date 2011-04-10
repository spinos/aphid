#include <QtGui>

#include <math.h>

#include "lbmSolver.h"

#define LAT_W 128
#define LAT_H 128
#define LAT_LEN 16384
#define idx(x,y) ((y)*LAT_W+(x))
#define M_WALL 0
#define M_XOUT 1
#define M_YOUT 2
#define M_XIN 3
#define M_YIN 4
#define M_FLUID 5
#define M_SETU 6

const float visc = 0.013f;

RenderThread::RenderThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
	
	tau = (5.f*visc + 1.0f)/2.0f;
	
	ux = new float[LAT_LEN];
	uy = new float[LAT_LEN];
	map = new short[LAT_LEN];
	//density = new float[LAT_LEN];
	pixel = new uchar[LAT_LEN*3];
	impulse_x = new float[LAT_LEN];
	impulse_y = new float[LAT_LEN];
		
	int x, y, i;
	
	for(i=0; i<9; i++)
	{
		lat[i] = new float[LAT_LEN];
	}

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {
			i = idx(x,y);
			map[i] = M_FLUID;
			lat[0][i] = 4.0f/9.0f;
			lat[1][i] =
			lat[2][i] =
			lat[3][i] =
			lat[4][i] = 1.0f/9.0f;
			lat[5][i] =
			lat[6][i] =
			lat[7][i] =
			lat[8][i] = 1.0f/36.0f;
		}
	}

	for (x = 0; x < LAT_W; x++) {
		map[idx(x,0)] = map[idx(x,LAT_H-1)] = M_WALL;
	}

	for (y = 0; y < LAT_H; y++) {
		map[idx(0,y)] = map[idx(LAT_W-1,y)] = M_WALL;
	}
	
	for (y = 1; y < LAT_H-1; y++) {
		map[idx(0,y)] = M_XIN;
		map[idx(LAT_W-1,y)] = M_XOUT;
	}
	
	_step = 0;
}

RenderThread::~RenderThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void RenderThread::render()
{
    QMutexLocker locker(&mutex);

	this->resultSize = QSize(128,128);
	

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
        mutex.lock();

        QSize resultSize = this->resultSize;
		
	simulate();
	simulate();
	simulate();
	simulate();
	
	mutex.unlock();
	
	

	_step++;
	
	for (int y = 0; y < LAT_H; ++y) 
	{
	if (restart)
                    break;
                if (abort)
                    return;
					
		
		for (int x = 0; x < LAT_W; ++x) 
		{
			int gi = idx(x, y);
			if(map[gi] != M_WALL)
			{
				int r = ux[gi]*128*4 + 128;
				if(r < 0) r = 0;
				else if(r > 255) r = 255;
				int g = uy[gi]*128*4 + 128;
				if(g < 0) g = 0;
				else if(g > 255) g = 255; 
			
				pixel[gi*3] = r;
				pixel[gi*3+1] = g;
				pixel[gi*3+2] = 128;
			}
			else
			{
				pixel[gi*3] = pixel[gi*3+1] = pixel[gi*3+2] = 0;
			}
		}
	}

        QImage image(pixel, LAT_W, LAT_H, QImage::Format_RGB888);
		
		if (!restart)
				emit renderedImage(image, _step);

		mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
    }
}

void RenderThread::getMacro(int x, int y, float &rho, float &vx, float &vy)
{
	int i;
	rho = 0.0;

	int gi = idx(x,y);

	if (map[gi] == M_FLUID) {
		for (i = 0; i < 9; i++) {
			rho += lat[i][gi];
		}
	
		vx = (lat[2][gi] + lat[5][gi] + lat[6][gi] - lat[8][gi] - lat[4][gi] - lat[7][gi])/rho;
		vy = (lat[1][gi] + lat[5][gi] + lat[8][gi] - lat[7][gi] - lat[3][gi] - lat[6][gi])/rho;
	} 
	else if (map[gi] == M_XIN) 
	{
		rho = 1.f;
		float dist_to_center = y - LAT_H/2;
		if(dist_to_center < 0) dist_to_center = -dist_to_center;
		
		dist_to_center /= LAT_H/2;
		
		vx = .17f * (1.f - dist_to_center*dist_to_center);
		vy =0.f;
	}
	else if (map[gi] == M_YIN) 
	{
		rho = 1.f;
		float dist_to_center = x - LAT_W/2;
		if(dist_to_center < 0) dist_to_center = -dist_to_center;
		
		dist_to_center /= LAT_W/2;
		
		vx =0.f;
		vy = .17f * (1.f - dist_to_center*dist_to_center);
		
	}
	else 
	{
	
		rho = 1.f;
			
		float decay = float(map[gi] - M_FLUID)/100.f;
		vx = impulse_x[gi] * decay;
		vy = impulse_y[gi] * decay;
		
		
	}
}

void RenderThread::simulate()
{
// relaxate
	int x, y, i, gi;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {

			gi = idx(x, y);
			
			if (map[gi] == M_WALL) {
				float tmp;
				tmp = lat[2][gi];
				lat[2][gi] = lat[4][gi];
				lat[4][gi] = tmp;

				tmp = lat[1][idx(x,y)];
				lat[1][gi] = lat[3][gi];
				lat[3][gi] = tmp;

				tmp = lat[8][idx(x,y)];
				lat[8][gi] = lat[6][gi];
				lat[6][gi] = tmp;

				tmp = lat[7][idx(x,y)];
				lat[7][gi] = lat[5][gi];
				lat[5][gi] = tmp;
			}
			else if(map[gi] == M_XOUT)
			{
				for (i = 0; i < 9; i++) {
						lat[i][gi] = lat[i][idx(x-1, y)];
				}
			}
			else if(map[gi] == M_YOUT)
			{
				for (i = 0; i < 9; i++) {
						lat[i][gi] = lat[i][idx(x, y-1)];
				}
			}
			else
			 {
				float v_x, v_y, rho;
				getMacro(x, y, rho, v_x, v_y);

				ux[gi] = v_x;
				uy[gi] = v_y;
				//density[gi] = rho;

				float Cusq = -1.5f * (v_x*v_x + v_y*v_y);
				float feq[9];

				feq[0] = rho * (1.0f + Cusq) * 4.0f/9.0f;
				feq[1] = rho * (1.0f + Cusq + 3.0f*v_y + 4.5f*v_y*v_y) / 9.0f;
				feq[2] = rho * (1.0f + Cusq + 3.0f*v_x + 4.5f*v_x*v_x) / 9.0f;
				feq[3] = rho * (1.0f + Cusq - 3.0f*v_y + 4.5f*v_y*v_y) / 9.0f;
				feq[4] = rho * (1.0f + Cusq - 3.0f*v_x + 4.5f*v_x*v_x) / 9.0f;
				feq[5] = rho * (1.0f + Cusq + 3.0f*(v_x+v_y) + 4.5f*(v_x+v_y)*(v_x+v_y)) / 36.0f;
				feq[6] = rho * (1.0f + Cusq + 3.0f*(v_x-v_y) + 4.5f*(v_x-v_y)*(v_x-v_y)) / 36.0f;
				feq[7] = rho * (1.0f + Cusq + 3.0f*(-v_x-v_y) + 4.5f*(v_x+v_y)*(v_x+v_y)) / 36.0f;
				feq[8] = rho * (1.0f + Cusq + 3.0f*(-v_x+v_y) + 4.5f*(-v_x+v_y)*(-v_x+v_y)) / 36.0f;

				if (map[gi] == M_FLUID) {
					for (i = 0; i < 9; i++) {
						lat[i][gi] += (feq[i] - lat[i][gi]) / tau;
					}
				} else {
					for (i = 0; i < 9; i++) {
						lat[i][gi] = feq[i];
					}
					if(map[gi] > M_FLUID) map[gi]--;
				}
			} 
		}
	}

// propagate

// west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = 0; y < LAT_H; y++) {
			lat[4][idx(x,y)] = lat[4][idx(x+1,y)];
		}
	}
	
	// east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = 0; y < LAT_H; y++) {
			lat[2][idx(x,y)] = lat[2][idx(x-1,y)];
		}
	}
	
	// north
	for (x = 0; x < LAT_W; x++) {
		for (y = LAT_H-1; y > 0; y--) {
			lat[1][idx(x,y)] = lat[1][idx(x,y-1)];
		}
	}

	// south
	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H-1; y++) {
			lat[3][idx(x,y)] = lat[3][idx(x,y+1)];
		}
	}

	// north-west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = LAT_H-1; y > 0; y--) {
			lat[8][idx(x,y)] = lat[8][idx(x+1,y-1)];
		}
	}

	// north-east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = LAT_H-1; y > 0; y--) {
			lat[5][idx(x,y)] = lat[5][idx(x-1,y-1)];
		}
	}

	// south-west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = 0; y < LAT_H-1; y++) {
			lat[7][idx(x,y)] = lat[7][idx(x+1,y+1)];
		}
	}

	// south-east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = 0; y < LAT_H-1; y++) {
			lat[6][idx(x,y)] = lat[6][idx(x-1,y+1)];
		}
	}

	
}

void RenderThread::addImpulse(int x, int y, float vx, float vy)
{
	int gi = idx(x,y);
	if (map[gi] != M_WALL) {
		map[gi] = M_SETU + 100;
		impulse_x[gi] = vx * 0.41f;
		impulse_y[gi] = vy * 0.41f;
	}
}

void RenderThread::addObstacle(int x, int y)
{
	int gi = idx(x,y);
	map[gi] = M_WALL;
}

