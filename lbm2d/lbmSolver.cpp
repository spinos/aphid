#include <QtGui>

#include <math.h>

#include "lbmSolver.h"

#define LAT_W 128
#define LAT_H 128
#define LAT_LEN 16384
#define idx(x,y) ((y)*LAT_W+(x))
#define M_FLUID 1
#define M_WALL 0
#define M_SETU 2

const float visc = 0.01f;

RenderThread::RenderThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
	
	tau = (4.f*visc + 1.0f)/2.0f;
	
	ux = new float[LAT_LEN];
	uy = new float[LAT_LEN];
	map = new short[LAT_LEN];
	density = new float[LAT_LEN];
		
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
	
	for (x = 63; x < 64; x++) {
		for (y = 110; y < 120; y++) {
			map[idx(x,y)] = M_SETU + 150;
			}
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
		
	simulate(0);
	simulate(0);
	simulate(0);
	simulate(1);
	
	
	_step++;

        mutex.unlock();

        QImage image(resultSize, QImage::Format_RGB32);
			
			for (int y = 0; y < resultSize.height(); ++y) 
			{
				 if (restart)
                    break;
                if (abort)
                    return;

                uint *scanLine = reinterpret_cast<uint *>(image.scanLine(y));
				for (int x = 0; x < resultSize.width(); ++x) 
				{
					int r = ux[idx(x, y)]*512 + 128;
					if(r < 0) r = 0;
					else if(r > 255) r = 255;
					int g = uy[idx(x, y)]*512 + 128;
					if(g < 0) g = 0;
					else if(g > 255) g = 255;
					*scanLine++ = qRgb(r, g, 128);
                }
            }

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

	for (i = 0; i < 9; i++) {
		rho += lat[i][gi];
	}

	if (map[gi] == M_FLUID) {
		vx = (lat[2][gi] + lat[5][gi] + lat[6][gi] - lat[8][gi] - lat[4][gi] - lat[7][gi])/rho;
		vy = (lat[1][gi] + lat[5][gi] + lat[8][gi] - lat[7][gi] - lat[3][gi] - lat[6][gi])/rho;
	} else {
		vx = 0.f;
		vy = -.41f * float(map[gi] - M_FLUID)/150.f;
	}
}

void RenderThread::simulate(char reset)
{
// relaxate
	int x, y, i, gi;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {

			gi = idx(x, y);

			if (map[idx(x,y)] > M_WALL) {
				float v_x, v_y, rho;
				getMacro(x, y, rho, v_x, v_y);

				ux[gi] = v_x;
				uy[gi] = v_y;
				density[gi] = rho;

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
					if(reset) map[gi]--;
				}
			} else {
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
		}
	}

// propagate

// west
	for (x = 0; x < LAT_W-1; x++) {
		for (y = 0; y < LAT_H; y++) {
			lat[4][idx(x,y)] = lat[4][idx(x+1,y)];
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

	// east
	for (x = LAT_W-1; x > 0; x--) {
		for (y = 0; y < LAT_H; y++) {
			lat[2][idx(x,y)] = lat[2][idx(x-1,y)];
		}
	}
	
	

}
