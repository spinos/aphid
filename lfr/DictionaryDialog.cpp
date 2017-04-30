#include <QtGui>
#include "DictionaryDialog.h"
#include "DictionaryView.h"
#include "LfMachine.h"
#include "AllMath.h"
#include <boost/format.hpp>

using namespace aphid;

namespace lfr {

DictionaryDialog::DictionaryDialog(LfMachine * world, QWidget *parent)
    : QDialog(parent)
{
	const LfParameter * lparam = world->param();
	int wd, ht;
	lparam->getDictionaryImageSize(wd, ht);
	
	std::string sprop = boost::str(boost::format("num atoms %1% atom size %2%") 
						% lparam->dictionaryLength() 
						% lparam->atomSize() );
	
	m_dictView = new DictionaryView(this);
	m_statistics = new QLabel(tr(sprop.c_str() ));
	
	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(m_dictView);
	layout->addWidget(m_statistics);
	layout->setStretch(0, 1);
	layout->setContentsMargins(4, 4, 4, 4);
	setLayout(layout);

    setWindowTitle(tr("Dictionary"));
	
	ClampInPlace<int>(wd, 100, 400);
	ClampInPlace<int>(ht, 100, 400);
	
    resize(wd, ht+10);
	
	connect(this, SIGNAL(sendDictionary(QImage)),
            m_dictView, SLOT(recvDictionary(QImage)));
}

void DictionaryDialog::recvDictionary(const QImage &image)
{
	if(isVisible() )
		emit sendDictionary(image);
}

}