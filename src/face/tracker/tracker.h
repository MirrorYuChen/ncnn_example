#ifndef _FACE_TRACKER_H_
#define _FACE_TRACKER_H_

#include <vector>
#include "../common/common.h"
#include "opencv2/core.hpp"

namespace mirror {
class Tracker {
public:
    Tracker();
    ~Tracker();
    int Track(const std::vector<FaceInfo>& curr_faces,
        std::vector<TrackedFaceInfo>* faces);

private:
    std::vector<TrackedFaceInfo> pre_tracked_faces_;
    const float minScore_ = 0.3f;
    const float maxScore_ = 0.5f;
};

}

#endif // !_FACE_TRACKER_H_
