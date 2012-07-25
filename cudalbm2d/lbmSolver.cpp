#include <QtGui>

#include <math.h>

#include "lbmSolver.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <solver_implement.h>

#define LAT_W 160
#define LAT_H 128
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
    checkCUDevice();
    
    restart = false;
    abort = false;
	
	tau = (5.f*visc + 1.0f)/2.0f;
	
	ux = new float[LAT_LEN];
	uy = new float[LAT_LEN];
	map = new short[LAT_LEN];
	density = new float[LAT_LEN];
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
			density[i] = 0.f;
		}
	}

	for (x = 0; x < LAT_W; x++) {
		map[idx(x,0)] = map[idx(x,LAT_H-1)] = M_WALL;
	}

	for (y = 0; y < LAT_H; y++) {
		map[idx(0,y)] = map[idx(LAT_W-1,y)] = M_WALL;
	}
	
	for (y = 1; y < LAT_H-1; y++) {
		map[idx(0,y)] = M_WALL;
		map[idx(LAT_W-1,y)] = M_WALL;
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

	_step++;
	
	for (int y = 0; y < LAT_H; ++y) 
	{
		if (restart)
                    break;
                if (abort)
                    return;
					
		int r, g, b = 128;
		for (int x = 0; x < LAT_W; ++x) 
		{
			int gi = idx(x, y);
			if(map[gi] != M_WALL)
			{
				
				/*r = ux[gi]*128*4 + 127;
				if(r < 0) r = 0;
				else if(r > 255) r = 255;
				g = uy[gi]*128*4 + 127;
				if(g < 0) g = 0;
				else if(g > 255) g = 255; 
				*/
				
				r = g = b = density[gi] * 127;
			
				pixel[gi*3] = r;
				pixel[gi*3+1] = g;
				pixel[gi*3+2] = b;
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

	//if (map[gi] >= M_FLUID) {
		for (i = 0; i < 9; i++) {
			rho += lat[i][gi];
		}
	
		vx = (lat[2][gi] + lat[5][gi] + lat[6][gi] - lat[8][gi] - lat[4][gi] - lat[7][gi])/rho;
		vy = (lat[1][gi] + lat[5][gi] + lat[8][gi] - lat[7][gi] - lat[3][gi] - lat[6][gi])/rho;
	//}
	/*
	else if (map[gi] == M_XIN) 
	{
		rho = 1.f;
		float dist_to_center = y - LAT_H/2;
		if(dist_to_center < 0) dist_to_center = -dist_to_center;
		
		dist_to_center /= LAT_H/2;
		
		vx = .13f * (1.f - dist_to_center);
		vx = .13f;
		vy =0.f;
	}
	else if (map[gi] == M_YIN) 
	{
		rho = 1.f;
		float dist_to_center = x - LAT_W/2;
		if(dist_to_center < 0) dist_to_center = -dist_to_center;
		
		dist_to_center /= LAT_W/2;
		
		vx =0.f;
		vy = .13f * (1.f - dist_to_center);
		
	}
	*/

}

void RenderThread::getForce(int gi, float &rho, float &vx, float &vy)
{
	float decay = float(map[gi] - M_SETU)/80.f;
	vx = impulse_x[gi] * decay;
	vy = impulse_y[gi] * decay;
	rho = decay;
}

void RenderThread::simulate()
{
	inject();
	boundaryConditions();
	propagate();
	collide();
	trasport();	
}

void RenderThread::inject()
{
	int x, y, i, gi;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {

			gi = idx(x, y);
			
			if (map[gi] > M_FLUID)
			{
				float v_x, v_y, rho;
				float feq[9];
					
//      f8  f1   f5
//           |   
//           |  
//           | 
//      f4---|--- f2
//           | 
//           |         and f0 for the rest (zero) velocity
//           |   
//      f7  f3   f6		
			
						getForce(gi, rho, v_x, v_y);
						density[gi] += rho;
						feq[1] = (0.f * v_x + 1.f * v_y)/9.f;
						feq[2] = (1.f * v_x + 0.f * v_y)/9.f;
						feq[3] = (0.f * v_x - 1.f * v_y)/9.f;
						feq[4] = (-1.f * v_x + 0.f * v_y)/9.f;
						feq[5] = (1.f * v_x + 1.f * v_y)/36.f;
						feq[6] = (1.f * v_x - 1.f * v_y)/36.f;
						feq[7] = (-1.f * v_x - 1.f * v_y)/36.f;
						feq[8] = (-1.f * v_x + 1.f * v_y)/36.f;
						
						for (i = 1; i < 9; i++) {
							lat[i][gi] += feq[i];
						}
												
						map[gi]--;
					

			} 
		}
	}
}

void RenderThread::boundaryConditions()
{
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
			 
		}
	}
}

void RenderThread::collide()
{
	// collide
	int x, y, i, gi;

	for (x = 0; x < LAT_W; x++) {
		for (y = 0; y < LAT_H; y++) {

			gi = idx(x, y);
			
			
				float v_x, v_y, rho;
				rho = lat[0][gi] + lat[1][gi] + lat[2][gi] + lat[3][gi] + lat[4][gi] + lat[5][gi] + lat[6][gi] + lat[7][gi] + lat[8][gi];
				
	
			v_x = (lat[2][gi] + lat[5][gi] + lat[6][gi] - lat[8][gi] - lat[4][gi] - lat[7][gi])/rho;
			v_y = (lat[1][gi] + lat[5][gi] + lat[8][gi] - lat[7][gi] - lat[3][gi] - lat[6][gi])/rho;
			
			v_y -= density[gi] * 0.0004f;
			float speedcap = 0.22f;
			if (v_x < -speedcap) v_x = -speedcap;
			if (v_x >  speedcap) v_x =  speedcap;
			if (v_y < -speedcap) v_y = -speedcap;
			if (v_y >  speedcap) v_y =  speedcap;
				ux[gi] = v_x;
				uy[gi] = v_y;
				

				
				float uu = v_x*v_x;
				float vv = v_y*v_y;
				float uv = v_x*v_y;
				float feq[9];
//      f8  f1   f5
//           |   
//           |  
//           | 
//      f4---|--- f2
//           | 
//           |         
//           |   
//      f7  f3   f6	

				feq[0] = rho * (1.0f - 1.5f * (uu + vv)) * 4.0f/9.0f;
				feq[1] = rho * (1.0f + 3.0f * v_y + 3.f * vv - 1.5f * uu) / 9.0f;
				feq[2] = rho * (1.0f + 3.0f * v_x + 3.f * uu - 1.5f * vv) / 9.0f;
				feq[3] = rho * (1.0f - 3.0f * v_y + 3.f * vv - 1.5f * uu) / 9.0f;
				feq[4] = rho * (1.0f - 3.0f * v_x + 3.f * uu - 1.5f * vv) / 9.0f;
				feq[5] = rho * (1.0f + 3.0f * v_x + 3.f * v_y + 3.f * uu + 3.f * vv + 9.f * uv) / 36.0f;
				feq[6] = rho * (1.0f + 3.0f * v_x - 3.f * v_y + 3.f * uu + 3.f * vv - 9.f * uv) / 36.0f;
				feq[7] = rho * (1.0f - 3.0f * v_x - 3.f * v_y + 3.f * uu + 3.f * vv + 9.f * uv) / 36.0f;
				feq[8] = rho * (1.0f - 3.0f * v_x + 3.f * v_y + 3.f * uu + 3.f * vv - 9.f * uv) / 36.0f;

				
					for (i = 0; i < 9; i++) {
						lat[i][gi] += (feq[i] - lat[i][gi]) / tau;
					}
					
		}
	}
}

void RenderThread::trasport()
{
	int x, y, gi;
	for (x = 1; x < LAT_W-1; x++) {
		for (y = 1; y < LAT_H-1; y++) {
			gi = idx(x, y);
			int x0 = x - ux[gi];
			int y0 = y - uy[gi];
			int x1 = x0 + 1;
			int y1 = y0 + 1;
			float fracx = x - ux[gi] - x0;
			float fracy = y - uy[gi] - y0;
			
			float mix0 = density[idx(x0, y0)] * (1.f - fracx) + density[idx(x1, y0)] * fracx;
			float mix1 = density[idx(x0, y1)] * (1.f - fracx) + density[idx(x1, y1)] * fracx;
			density[gi] = mix0 * (1.f - fracy) + mix1 * fracy;
			
			if(density[gi] < 0.f) density[gi] = 0.f;
			if(density[gi] > 2.f) density[gi] = 2.f;
		}
	}
}

void RenderThread::propagate()
{
// propagate (stream)
	int x, y;

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
	if (map[gi] >= M_FLUID) {
		if(map[gi] == M_FLUID) map[gi] = M_SETU + 40;
		impulse_x[gi] = vx;
		impulse_y[gi] = vy;
	}
}

void RenderThread::addObstacle(int x, int y)
{
	int gi = idx(x,y);
	map[gi] = M_WALL;
}


