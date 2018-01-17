// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;

void test(SupervisedLearner& learner, const char* challenge)
{
	// Load the training data
	string fn = "data/";
	fn += challenge;
	Matrix trainFeatures;
	trainFeatures.loadARFF(fn + "_train_feat.arff");
	Matrix trainLabels;
	trainLabels.loadARFF(fn + "_train_lab.arff");

	// Train the model
	learner.train(trainFeatures, trainLabels);

	// Load the test data
	Matrix testFeatures;
	testFeatures.loadARFF(fn + "_test_feat.arff");
	Matrix testLabels;
	testLabels.loadARFF(fn + "_test_lab.arff");

	// Measure and report accuracy
	size_t misclassifications = learner.countMisclassifications(testFeatures, testLabels);
	cout << "Misclassifications by " << learner.name() << " at " << challenge << " = " << to_str(misclassifications) << "/" << to_str(testFeatures.rows()) << "\n";
}

void testLearner(SupervisedLearner& learner)
{
	test(learner, "hep");
	test(learner, "vow");
	test(learner, "soy");
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
		BaselineLearner bl;
		testLearner(bl);
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}
