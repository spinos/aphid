/*
 *  ImageView.h
 *  mallard
 *
 *  Created by jian zhang on 12/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseServer.h>
#include <QWidget>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

class Qimage;
class BaseServer;

class ImageView : public QWidget, public BaseServer
{
    Q_OBJECT

public:
    ImageView(QWidget *parent = 0);
	void resizeImage(QSize s);
	void setRendererName(QString name);
	void startRender(QString name);
protected:
	virtual void processRead(const char * data, size_t length);
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    
private:
	void beginBucket(const char * data);
	void processPackage(const char * data);
	void endBucket();
	void doProcessPackage(const char * data);
	void doEndBucket();
	void showImageName(QPainter & painter);
	void showImageSize(QPainter & painter);
	void showRenderer(QPainter & painter);
	void showStatus(QPainter & painter);
	void showRenderTime(QPainter & painter);
	void showBucket(QPainter & painter);
	QString statusString() const;
	QString renderTimeString();
	int renderTimeInt();
private:
	enum RenderStatus {
		Idle = 0,
		Active = 1,
		Finished = 2
	};
	
	boost::posix_time::ptime m_renderBeginTime;
	QString m_imageName;
	QString m_rendererName;
    QImage * m_image;
	float * m_colors;
	int bucketRect[4];
	int m_packageStart;
	unsigned numPix;
	int m_numBucket, m_numFinishedBucket, m_renderTimeSecs;
	RenderStatus m_status;
};