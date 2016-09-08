#include <QtGui>
#include "ImageDialog.h"
#include "DictionaryView.h"
#include "LfMachine.h"
#include "AllMath.h"
#include <boost/format.hpp>

using namespace aphid;

namespace lfr {

ImageDialog::ImageDialog(LfMachine * world, QWidget *parent)
    : QDialog(parent)
{
	int wd = 100, ht = 100;
	std::string sprop = "unknown";
	std::string stitle = "Unknown Image";
	
	const LfParameter * lparam = world->param();
	if(lparam->numImages() > 0) {
		lparam->getImageSize(wd, ht, 0);
		sprop = boost::str(boost::format("num pixels %1%") 
						% lparam->imageNumPixels(0) );
		stitle = lparam->imageName(0);
	}
	
	m_dictView = new DictionaryView(this);
	m_statistics = new QLabel(tr(sprop.c_str() ));
	
	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(m_dictView);
	layout->addWidget(m_statistics);
	layout->setStretch(0, 1);
	layout->setContentsMargins(4, 4, 4, 4);
	setLayout(layout);

    setWindowTitle(tr(stitle.c_str()));
	
    resize(wd, ht+10);
	
	connect(this, SIGNAL(sendImage(QImage)),
            m_dictView, SLOT(recvDictionary(QImage)));
}

void ImageDialog::recvImage(const QImage &image)
{
	if(isVisible() )
		emit sendImage(image);
}

}