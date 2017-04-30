#include <QApplication>
#include <QtCore>
#include "dctmn.h"
#include "window.h"

using namespace lfr;

int main(int argc, char *argv[])
{
#if 1
	LfParameter param(argc, argv);
	
	if(!param.isValid()) {
		param.printHelp();
		return 1;
	}
	
	int nt = param.numThread();
	if(nt < 1) nt = 1;
	if(nt > 24) nt = 24;
	qDebug()<<" using "<<nt<<" threads  ";
    
	LfMachine * machine;
	switch(nt) {
	    case 1 :
	        machine = (LfMachine *)(new DictionaryMachine<1, float>(&param));
	    break;
	    case 3 :
	        machine = (LfMachine *)(new DictionaryMachine<3, float>(&param));
	    break;
	    case 4 :
	        machine = (LfMachine *)(new DictionaryMachine<4, float>(&param));
	    break;
	    case 5 :
	        machine = (LfMachine *)(new DictionaryMachine<5, float>(&param));
	    break;
	    case 6 :
	        machine = (LfMachine *)(new DictionaryMachine<6, float>(&param));
	    break;
	    case 7 :
	        machine = (LfMachine *)(new DictionaryMachine<7, float>(&param));
	    break;
	    case 8 :
	        machine = (LfMachine *)(new DictionaryMachine<8, float>(&param));
	    break;
	    case 9 :
	        machine = (LfMachine *)(new DictionaryMachine<9, float>(&param));
	    break;
	    case 10 :
	        machine = (LfMachine *)(new DictionaryMachine<10, float>(&param));
	    break;
	    case 11 :
	        machine = (LfMachine *)(new DictionaryMachine<11, float>(&param));
	    break;
	    case 12 :
	        machine = (LfMachine *)(new DictionaryMachine<12, float>(&param));
	    break;
	    case 13 :
	        machine = (LfMachine *)(new DictionaryMachine<13, float>(&param));
	    break;
	    case 14 :
	        machine = (LfMachine *)(new DictionaryMachine<14, float>(&param));
	    break;
	    case 15 :
	        machine = (LfMachine *)(new DictionaryMachine<15, float>(&param));
	    break;
	    case 16 :
	        machine = (LfMachine *)(new DictionaryMachine<16, float>(&param));
	    break;
	    case 17 :
	        machine = (LfMachine *)(new DictionaryMachine<17, float>(&param));
	    break;
	    case 18 :
	        machine = (LfMachine *)(new DictionaryMachine<18, float>(&param));
	    break;
	    case 19 :
	        machine = (LfMachine *)(new DictionaryMachine<19, float>(&param));
	    break;
	    case 20 :
	        machine = (LfMachine *)(new DictionaryMachine<20, float>(&param));
	    break;
	    case 21 :
	        machine = (LfMachine *)(new DictionaryMachine<21, float>(&param));
	    break;
	    case 22 :
	        machine = (LfMachine *)(new DictionaryMachine<22, float>(&param));
	    break;
	    case 23 :
	        machine = (LfMachine *)(new DictionaryMachine<23, float>(&param));
	    break;
	    case 24 :
	        machine = (LfMachine *)(new DictionaryMachine<24, float>(&param));
	    break;
	    default :
	        machine = (LfMachine *)(new DictionaryMachine<2, float>(&param));
	    break;
	}
	
	
    QApplication app(argc, argv);
    Window window(machine);
	//window.showMaximized();
    window.resize(800, 600);
    window.show();
    return app.exec();
#else
	LfWorld::testLAR();
    return 1;
#endif
}
