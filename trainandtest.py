import pickle
import sys

from DataSet import DataSet
from HMM import HMM

if __name__ == "__main__":
    """Read in data, call code to train HMM, and save model."""

    # This function should be called with one argument: trainingdata.txt

    dataset = DataSet(r"./data/randomwalk.train.txt")
    dataset.readFile()

    hmm = HMM(dataset.envShape)
    hmm.train(dataset.observations)

    # Save the model for future use
    fileName = "models/trained-model.pkl"
    print("Saving trained model as " + fileName)
    pickle.dump(
        {"T": hmm.transition_probs, "M": hmm.emission_probs, "pi": hmm.start_probs},
        open(fileName, "wb"),
    )

    """Call Viterbi implementation of HMM on a given set of observations."""
    # This function with the test data and (optionally) a model file

    dataset = DataSet(r"./data/randomwalk.test.txt")
    dataset.readFile()

    totalCorrect = 0
    totalIncorrect = 0
    for i in range(len(dataset.observations)):
        print(f"obs{i}")
        predictedStates = hmm.viterbi(dataset.observations[i])
        print(predictedStates[:20])
        print(dataset.states[i][:20])
        if len(predictedStates) != len(dataset.states[i]):
            print("Length of predictedStates differs from dataset.states")
            sys.exit(-1)
        trueStates = dataset.states[i]

        numCorrect = 0
        for j in range(len(dataset.states[i])):
            if predictedStates[j] == dataset.states[i][j]:
                numCorrect += 1

        totalCorrect += numCorrect
        totalIncorrect += len(dataset.observations[i]) - numCorrect
        print()

    accuracy = totalCorrect / (totalCorrect + totalIncorrect)
    print("Accuracy: {0:.2f} percent".format(100 * accuracy))
