//
// Created by Oğuz Kerem Yıldız on 3.12.2025.
//

#ifndef CLASSIFICATION_SOFTDECISIONTREEPARAMETER_H
#define CLASSIFICATION_SOFTDECISIONTREEPARAMETER_H
#include "LinearPerceptronParameter.h"


class SoftDecisionTreeParameter : public LinearPerceptronParameter {
private:
    double alpha;
public:
    SoftDecisionTreeParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, double alpha, int n, int p, int id);
    double getAlpha() const;
    int n;
    int p;
    int id;
};


#endif //CLASSIFICATION_SOFTDECISIONTREEPARAMETER_H