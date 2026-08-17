//
// Created by Oğuz Kerem Yıldız on 3.12.2025.
//

#include "SoftDecisionTreeParameter.h"

SoftDecisionTreeParameter::SoftDecisionTreeParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, double alpha, int n, int p, int id) : LinearPerceptronParameter(seed, learningRate, etaDecrease, crossValidationRatio, epoch) {
    this->alpha = alpha;
    this->n = n;
    this->p = p;
    this->id = id;
}

double SoftDecisionTreeParameter::getAlpha() const {
    return alpha;
}
