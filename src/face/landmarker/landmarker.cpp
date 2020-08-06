#include "landmarker.h"
#include "zqlandmarker/zqlandmarker.h"
#include "insightface/insightface.h"

namespace mirror {
Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
	return new ZQLandmarker();
}

Landmarker* InsightfaceLandmarkerFactory::CreateLandmarker() {
	return new InsightfaceLandmarker();
}

}
