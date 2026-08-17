//
// Created by Oğuz Kerem Yıldız on 3.12.2025.
//

#include "catch.hpp"
#include "../src/DataSet/DataSet.h"
#include "../src/Model/DecisionTree//DecisionTree.h"
#include "../src/Model/DecisionTree/SoftDecisionTree.h"
#include "../src/Parameter/SoftDecisionTreeParameter.h"

TEST_CASE("SoftTreeTest") {
    DataSet iris, bupa, dermatology;
    vector<AttributeType> attributeTypes;
    vector<vector<string>> attributeValueList;
    for (int i = 0; i < 4; i++) {
        attributeTypes.emplace_back(AttributeType::CONTINUOUS);
    }
    DataDefinition dataDefinition = DataDefinition(attributeTypes);
    iris = DataSet(dataDefinition, ",", "datasets/iris.data");
    attributeTypes.clear();
    for (int i = 0; i < 6; i++) {
        attributeTypes.emplace_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    bupa = DataSet(dataDefinition, ",", "datasets/bupa.data");
    attributeTypes.clear();
    for (int i = 0; i < 34; i++) {
        attributeTypes.emplace_back(AttributeType::CONTINUOUS);
    }
    dataDefinition = DataDefinition(attributeTypes);
    dermatology = DataSet(dataDefinition, ",", "datasets/dermatology.data");
    SECTION("train"){
        SoftDecisionTreeParameter* softTreeParameter;
        softTreeParameter = new SoftDecisionTreeParameter(1, 0.1, 0.99, 0.2, 100, 0, 16, 1, 0);
        auto softTree = SoftDecisionTree();
        InstanceList instanceList = dermatology.getInstanceList();
        softTree.train(instanceList, softTreeParameter);
        REQUIRE_THAT(2.73, Catch::Matchers::WithinAbs(100 * softTree.test(dermatology.getInstanceList())->getErrorRate(), 0.01));
        softTree = SoftDecisionTree();
        instanceList = iris.getInstanceList();
        softTreeParameter = new SoftDecisionTreeParameter(1, 0.1, 0.99, 0.2, 10, 0.2, 16, 1, 0);
        softTree.train(instanceList, softTreeParameter);
        REQUIRE_THAT(4.66, Catch::Matchers::WithinAbs(100 * softTree.test(iris.getInstanceList())->getErrorRate(), 0.01));
        softTreeParameter = new SoftDecisionTreeParameter(1, 0.001, 0.99, 0.2, 100, 0, 16, 1, 0);
        softTree = SoftDecisionTree();
        instanceList = bupa.getInstanceList();
        softTree.train(instanceList, softTreeParameter);
        REQUIRE_THAT(26.95, Catch::Matchers::WithinAbs(100 * softTree.test(bupa.getInstanceList())->getErrorRate(), 0.01));
    }
}