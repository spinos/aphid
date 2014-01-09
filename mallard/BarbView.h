#pragma once
#include <Base3DView.h>
#include <FeatherExample.h>
#include <LODFn.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "FeatherAttrib.h"
class MlFeather;
class BarbView : public Base3DView, public FeatherExample, public LODFn, public FeatherAttrib {
Q_OBJECT

public:
    BarbView(QWidget *parent = 0);
    ~BarbView();
	
protected:
	virtual void clientSelect();
	virtual void clientDeselect();
	virtual void clientDraw();
	virtual void clientMouseInput();
	virtual void focusInEvent(QFocusEvent * event);
	
public slots:
	void receiveSeed(int s);
	void receiveNumSeparate(int n);
	void receiveSeparateStrength(double k);
	void receiveFuzzy(double f);
	void receiveResShaft(int g);
	void receiveResBarb(int g);
	void receiveLod(double l);
	void receiveShapeChanged();
	
	void test();
private:
    void sampleShape();
    MlFeather * m_f;
    boost::mutex io_mutex;
};
