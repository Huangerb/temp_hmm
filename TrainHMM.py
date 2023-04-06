import pickle
import sys

from DataSet import DataSet
from HMM import HMM

if __name__ == "__main__":
    """Read in data, call code to train HMM, and save model."""

    # This function should be called with one argument: trainingdata.txt
    if len(sys.argv) != 2:
        print("Usage: TrainMM.py trainingdata.txt")
        sys.exit(0)

    dataset = DataSet(sys.argv[1])
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
