

from Creatures import *
import copy





class EvolveStrategy(object):




    def cross(self,cross_creature_pair_list, cross_result_num_each):
        '''
        :param cross_group_list:
        :param cross_result_num_each: the num of Creatures cross will get for each pair of Cross group
        '''

        new_creatures = []
        for cross_pair in cross_creature_pair_list:
            for i in range(cross_result_num_each):
                new_creatures.append(Brain.surgery_mix_brain_weights_get_creatures([c.brain for c in cross_pair]))

        return new_creatures


    def mutate(self,mutate_creature_list, mutate_result_num_each,mutate_std):

        new_creatures = []
        for creature in mutate_creature_list:
            for i in range(mutate_result_num_each):
                aim_creature = creature.copy()
                aim_creature.brain.weights_add_gaussian(mutate_std)
                new_creatures.append(aim_creature)
        for i in new_creatures:
            i.score = 0
        return new_creatures


    def evolve(self,group):
        raise  NotImplementedError



class HighScoreKeepStrategy(EvolveStrategy):


    def evolve(self,group):
        self.creatures = group.sort_by_score()
        group_number = len(self.creatures)

        keep_num = 1
        mutate_num = int(0.3 * group_number) + 1
        cross_num = int(0.2 * group_number) + 1
        output = []
        for c in self.creatures[:keep_num]:
            output.append(c.copy())


        mutate_1 = self.mutate(self.creatures[:mutate_num],1,0.1)
        mutate_2 = self.mutate(self.creatures[:mutate_num],1,0.2)
        cross = self.cross([self.creatures[:cross_num]],cross_num)

        output.extend(mutate_1)
        output.extend(mutate_2)
        output.extend(cross)


        return output












class TimeStrategy(EvolveStrategy):
    '''
    e.g. every 1 minute, 2 minute, 3 minute... , evolve the live creatures, the creature with longest life keeps remain
    '''
    pass
