#ifndef LfThread_H
#define LfThread_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

namespace lfr {

class LfMachine;

class LfThread : public QThread
{
    Q_OBJECT

public:
    LfThread(LfMachine * world, QObject *parent = 0);
    virtual ~LfThread();

    void render(QSize resultSize);
	void initAtoms();
	void beginLearn();
	
signals:
	void sendInitialDictionary(const QImage &image);
	void sendDictionary(const QImage &image);
	void sendSparsity(const QImage &image);
    void sendPSNR(float value);

protected:
    void run();

private:
	LfMachine * m_world;
    QMutex mutex;
    QWaitCondition condition;

    QSize resultSize;
    bool restart;
    bool abort;

	QImage * m_spasityImg;
	QImage * m_dictImg;
};

}
#endif
