# Import packages
import copy

# Memory
class Memory:

    def __init__(self, config):

        self._experience = []
        self._experiences = []
        self._experience_buffer = []
        self._config = config

    def update_experience(self, options, state):

        """
        Update remaining experience with next state and options for next state
        """

        if state is None:
            self._experience += [None, [False, False, False, False, False, False, False, False, True]]
        if len(self._experience) == 3:
            self._experience += [state, [bool(o) for o in options]]

    def add_corrected_experience(self, tag_happened):

        """
        Add experience to experience history (and fix buffer if necessary)
        The experience buffer is there in case the player is playing against itself. Then the next state is not the next
        experience, but the one where it is in the same role again.
        """

        # If the experience is empty, you're done
        if len(self._experience) == 0:
            return None

        # If no tag happened, check if experience needs correction (if yes add to experience buffer)
        if tag_happened:
            if self._experience[3] is not None:
                self._experience[3:5] = (None, [False, False, False, False, False, False, False, False, True])
            self._experience_buffer += [copy.deepcopy(self._experience)]
            self.add_experience()
        else:
            # If role of current and next state are different
            # due to player playing against itself: buffer to correct
            if self._experience[0][-2] is not self._experience[3][-2]:
                self._experience_buffer += [copy.deepcopy(self._experience)]
                self._experience = []
            else:
                self.add_experience()

        # For the experience buffer, find the subsequent experiences with matching roles
        # The state and next state should have the same role for one experience
        self.correct_experience_buffer()

        # If the amount of experiences exceeds memory, truncate
        if len(self._experiences) > self._config.memory_size:
            self._experiences = self._experiences[-self._config.memory_size:]

    def add_experience(self):
        self._experiences += [self._experience]
        self._experience = []

    def correct_experience_buffer(self):

        """
        Take the first experience in the buffer, then find the matching experience to correct the first.
        The experience buffer is there in case the player is playing against itself. Then the next state is not the next
        experience, but the one where it is in the same role again.
        """

        # Take the first experience
        i = 0; del_is = []

        while len(self._experience_buffer) > (i + 1):
            experience_to_correct = copy.deepcopy(self._experience_buffer[i])
            turn = experience_to_correct[0][-2]
            match_found = False
            for j, _experience in enumerate(self._experience_buffer[(i+1):], start=i+1):

                # Find matching experience (same role)
                if _experience[3] is None and _experience[0][-2] == turn:
                    experience_to_correct[-2] = _experience[0]
                    experience_to_correct[-1] = _experience[-1]
                    match_found = True

                elif len(_experience) == 5 and _experience[3] is not None and _experience[3][-2] == turn:

                    # Take this found experience to correct the next state and next options of the experience to correct
                    experience_to_correct[-2:] = _experience[-2:]
                    match_found = True

                if match_found:
                    self._experience = experience_to_correct
                    self.add_experience()

                    # Once corrected, it can be deleted from the buffer
                    if i not in del_is:
                        del_is += [i]
                    break

            # Delete from buffer
            self._experience_buffer = [s for d, s in enumerate(self._experience_buffer) if d not in del_is]
            i+=1; i-=len(del_is); del_is = []

        # Clean up the finished experiences
        self._experience_buffer = [s for s in self._experience_buffer if s[3] is not None]

    @property
    def experience(self):
        return self._experience

    @experience.setter
    def experience(self, values):
        if len(values) == 3:
            choice, reward, state = values
            self._experience = [state, int(choice), reward]
        elif len(values) == 5:
            self._experience = values
        else:
            raise ValueError(
                "The experience setter expects either:"
                " - a tuple of three values: (choice, reward, state); or"
                " - a list of all five experience values: [state, choice, reward, next_state, next_options]"
            )


    @property
    def experiences(self):
        return self._experiences