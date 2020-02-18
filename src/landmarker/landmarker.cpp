#include "landmarker.h"
#include "zqlandmark/zq_landmarker.h"

namespace mirror {
Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
	return new ZQLandmarker();
}

}
