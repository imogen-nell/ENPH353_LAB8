import random
import csv
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        self.q = pickle.load(open(filename + '.pkl', 'rb'))
        print("Loaded file: {}".format(filename + ".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        # Save Q values to a pickle file
        pickle.dump(self.q, open(filename + '.pkl', 'wb'))
        print("Wrote to file: {}".format(filename+".pickle"))
        # Save Q values to a CSV file
        with open(filename + ".csv", 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for (state, action) in self.q:
                csv_writer.writerow([state, action, self.q.get((state,action))])

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value? TODO
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action ???
 
        ##0 - explore, 1 - exploit
        ##choses weiather to explore or explot with epsilon chance of exploration
        exploreOrExploit = random.choices([0, 1], [self.epsilon, 1 - self.epsilon])[0]
        exploreChoice = random.choice([0, 1, 2])        
        if (exploreOrExploit==0):
            #explore
            return self.actions[exploreChoice]
        else:
            #exploit
            highestValueReward = max([self.getQ(state, a) for a in self.actions])
            actionMax = [a for a in self.actions if self.getQ(state, a) == highestValueReward]
            return random.choice(actionMax)
    


    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        
        # Find Q for current (state1, action1)
        Q_sa = self.getQ(state1, action1)

        # Find max(Q) for state2
        max_Q_s2 = max([self.getQ(state2, a) for a in self.actions])

        # Update the Q-value in the Q-table
        self.q[(state1, action1)] = Q_sa + self.alpha * (reward + self.gamma * max_Q_s2 - Q_sa)

