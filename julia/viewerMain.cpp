/* 
 *
 */
 
#include <QApplication>
#include "viewerWindow.h"
#include "CubeRender.h"
#include "AssetRender.h"
#include "viewerParameter.h"

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
		
		aphid::AssetRender * ar = new aphid::AssetRender;
		if( ar->load(param.inFileName(), param.assetGridLevel() ) )
			r = ar;
	}
	
	if(!r) {
		std::cout<<"\n no render attached ";
		return 0;
	}
	
	QApplication app(argc, argv);
	jul::Window window(r, param.operationTitle() );
    //window.showMaximized();
    window.resize(720, 540);
    window.show();
	return app.exec();
}
