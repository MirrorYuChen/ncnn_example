#include "landmarker.h"
#include "zqlandmarker/zqlandmarker.h"

namespace mirror {
Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
	return new ZQLandmarker();
}

}
