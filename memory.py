# Import packages
import copy
from config import Config
import numpy as np

# Memory
class Memory(Config):

    def __init__(self, maxMemory):

        # Import config
        Config.__init__(self)

        self._sample = []
        self._sample_many = []
        self._samples = []
        self._sample_buffer = []
        self._samples_count = 0
        self.maxMemory = maxMemory


    def get_sample(self):
        return self._sample

    sample = property(get_sample)

    def set_sample(self, values):
        choice, reward, state = values
        self._sample = [state, choice, reward]

    sample = property(get_sample, set_sample)

    def get_samples(self):
        return self._samples

    samples = property(get_samples)

    def set_samples(self, samples):
        self._samples = samples

    samples = property(get_samples, set_samples)

    def update_sample(self, options, state):

        """
        Update remaining sample with next state and options for next state
        """

        if options is None:
            self._sample += [None, None]
        if len(self._sample) == 3:
            options = np.where(options)[0].tolist()
            self._sample += [state, list(options)]

    def add_corrected_sample(self, tag_happened):

        """
        Add sample to sample history (and fix buffer if necessary)
        The sample buffer is there in case the player is playing against itself. Then the next state is not the next
        sample, but the one where it is in the same role again.
        """

        # If the sample is empty, you're done
        if len(self._sample) == 0:
            return None

        # If no tag happened, check if sample needs correction (if yes add to sample buffer)
        if tag_happened:
            if self._sample[3] is not None:
                self._sample[3:5] = (None, None)
            self._sample_buffer += [copy.deepcopy(self._sample)]
            self.add_sample()
        else:
            # If role of current and next state are different
            # due to player playing against itself: buffer to correct
            if self._sample[0][-2] is not self._sample[3][-2]:
                self._sample_buffer += [copy.deepcopy(self._sample)]
                self._sample = []
            else:
                self.add_sample()

        # For the sample buffer, find the subsequent samples with matching roles
        # The state and next state should have the same role for one sample
        self.correct_sample_buffer()

        # If the amount of samples exceeds memory, truncate
        if len(self._samples) > self.maxMemory:
            self._samples = self._samples[-self.maxMemory:]

    def add_sample(self):
        self._samples_count += 1
        self._samples += [self._sample]
        self._sample = []

    def correct_sample_buffer(self):

        """
        Take the first sample in the buffer, then find the matching sample to correct the first.
        The sample buffer is there in case the player is playing against itself. Then the next state is not the next
        sample, but the one where it is in the same role again.
        """

        # Take the first sample
        i = 0; del_is = []
        while len(self._sample_buffer) > (i + 1):
            sample_to_correct = copy.deepcopy(self._sample_buffer[i])
            turn = sample_to_correct[0][-2]
            match_found = False
            for j, _sample in enumerate(self._sample_buffer[(i+1):], start=i+1):

                # Find matching sample (same role)
                if _sample[3] is None and _sample[0][-2] == turn:
                    sample_to_correct[-2] = _sample[0]
                    sample_to_correct[-1] = [_sample[1]]
                    match_found = True

                elif len(_sample) is 5 and _sample[3] is not None and _sample[3][-2] == turn:

                    # Take this found sample to correct the next state and next options of the sample to correct
                    sample_to_correct[-2:] = _sample[-2:]
                    match_found = True

                if match_found:
                    self._sample = sample_to_correct
                    self.add_sample()

                    # Once corrected, it can be deleted from the buffer
                    if i not in del_is:
                        del_is += [i]
                    break

            # Delete from buffer
            self._sample_buffer = [s for d, s in enumerate(self._sample_buffer) if d not in del_is]
            i+=1; i-=len(del_is); del_is = []

        # Clean up finished samples
        self._sample_buffer = [s for s in self._sample_buffer if s[3] is not None]