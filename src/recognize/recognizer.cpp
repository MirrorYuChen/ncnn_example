#include "recognizer.h"
#include "./mobilefacenet/mobilefacenet.h"

namespace mirror {
Recognizer* MobilefacenetRecognizerFactory::CreateRecognizer() {
	return new Mobilefacenet();
}

}
