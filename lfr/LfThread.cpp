#include <QtGui>
#include <math.h>
#include "LfThread.h"

LfThread::LfThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;

}

LfThread::~LfThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void LfThread::render(QSize resultSize)
{
    QMutexLocker locker(&mutex);

	this->resultSize = resultSize;

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        condition.wakeOne();
    }
}

void LfThread::run()
{
    forever {
        mutex.lock();

        QSize resultSize = this->resultSize;

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
					int g = random()%256;
					*scanLine++ = qRgb(g, g, g);
                }
            }

			if (!restart)
				emit renderedImage(image);

        mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
    }
}

