#include "FeatherExample.h"
#include "MlFeather.h"
#include "MlFeatherCollection.h"

MlFeatherCollection * FeatherExample::FeatherLibrary = 0;
FeatherExample::FeatherExample() 
{
}

MlFeather * FeatherExample::selectedExample() const
{
	if(!FeatherLibrary) return 0;
	return FeatherLibrary->selectedFeatherExample();
}