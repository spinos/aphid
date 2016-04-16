/* 
 *
 */
 
#include <QApplication>
#include "viewerParameter.h"
#include "viewerWindow.h"
#include "CubeRender.h"
#include "WorldRender.h"

int main(int argc, char *argv[])
{	
	jul::ViewerParam param(argc, argv);
	if(!param.isValid() || param.operation() == jul::ViewerParam::kHelp ) {
		jul::ViewerParam::PrintHelp();
		return 0;
	}
	
	aphid::CudaRender * r = NULL;
	
	if(param.operation() == jul::ViewerParam::kTestVoxel ) {
		
		r = new aphid::CubeRender;
	}
	
	if(param.operation() == jul::ViewerParam::kTestAsset ) {
		
		// r = new aphid::WorldRender(param.inFileName() );
	}
	
	if(!r) {
		std::cout<<"\n no render ";
		return 0;
	}
	
	QApplication app(argc, argv);
	jul::Window window(r, param.operationTitle() );
    //window.showMaximized();
    window.resize(720, 540);
    window.show();
	return app.exec();
}
