#include <QtGui>

#include "QIconFrame.h"

namespace aphid {

QIconFrame::QIconFrame(QWidget *parent)
    : QLabel(parent)
{
	currentIconIndex = 0;
	setMinimumSize(32,32);
}

void QIconFrame::addIconFile(const QString & fileName)
{
	QPixmap *pix = new QPixmap(fileName);
	icons << pix;
}

void QIconFrame::setIconIndex(int index)
{
	currentIconIndex = index;
	if(currentIconIndex >= icons.size())
		currentIconIndex = 0;
	setPixmap(*(icons.at(currentIconIndex)));
}

int QIconFrame::getIconIndex() const
{
	return currentIconIndex;
}

char QIconFrame::useNextIcon()
{
	if(icons.size() < 1)
		return 0;
		
	setIconIndex(currentIconIndex+1);
	return 1;
}

void QIconFrame::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
		useNextIcon();
		
		QPair<int, int> val;
		val.first = m_nameId;
		val.second = currentIconIndex;
		
		emit iconChanged2(val);
	}
}

void QIconFrame::mouseReleaseEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {	
	}
}

void QIconFrame::setNameId(int x)
{ m_nameId = x; }
	
const int& QIconFrame::nameId() const
{ return m_nameId; }

}
