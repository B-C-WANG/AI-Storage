from EvolveStrategy import *
import copy
import numpy as np
import matplotlib.pylab as p

DEAD_ZONE = 3
PLAYER = 1
GROUND = 0
GOAL = 2
WALL = 4

class SingleCreatureDemoParams():
    def __init__(self):


        self.space_size = [5,5]
        self.init_player_pos = [np.random.randint(1,self.space_size[0]),
                      np.random.randint(1, self.space_size[1])]

        self.goal_pos = [np.random.randint(1,self.space_size[0]),
                      np.random.randint(1, self.space_size[1])]

        self.dead_zone_pos = [np.random.randint(1,self.space_size[0]),
                      np.random.randint(1, self.space_size[1])]



        while self.init_player_pos == self.goal_pos or self.goal_pos == self.dead_zone_pos or self.init_player_pos == self.dead_zone_pos:
            print("change")
            self.goal_pos = [np.random.randint(1, self.space_size[0]),
                             np.random.randint(1, self.space_size[1])]

            self.dead_zone_pos = [np.random.randint(1, self.space_size[0]),
                                  np.random.randint(1, self.space_size[1])]


class SingleCreatureDemo():
    '''
    loop: give observations to creature, get action, change states
    finally: gameover and get reward

    SingleCreatureDemo: Only One creature will be in each environment!

    '''

    def __init__(self, params, creature=None):
        print(type(creature))
        assert isinstance(creature, Creatures)
        assert isinstance(params, SingleCreatureDemoParams)
        self.creature = creature
        self.params = params
        self.__set_environment()
        self.__set_init_object_pos()
        self.game_over = 0
        self.step = 0

        self.last_distance = 999

    def __set_init_object_pos(self):

        self.__set_point_in_environment(self.params.init_player_pos, PLAYER)
        self.player_pos = self.params.init_player_pos
        self.__set_point_in_environment(self.params.dead_zone_pos,DEAD_ZONE)
        self.dead_zone_pos = self.params.dead_zone_pos
        self.__set_point_in_environment(self.params.goal_pos,GOAL)
        self.goal_pos = self.params.goal_pos


    def __set_environment(self):

        # set ground
        self.environment = np.zeros(shape=np.array(self.params.space_size)+1)
        # set border
        self.environment[0,:] = WALL
        self.environment[:,0] = WALL
        self.environment[-1, :] = WALL
        self.environment[:, -1] = WALL

    def __set_point_in_environment(self,list,value):
        assert len(list) == 2
        self.environment[list[0], list[1]] = value


    def get_object_in_space(self,pos):
        return self.environment[pos[0],pos[1]]

    def set_player_pos(self,pos):
        self.__set_point_in_environment(self.player_pos,GROUND)
        self.__set_point_in_environment(pos,PLAYER)
        self.player_pos = pos

    def run_output_observations_input_actions(self):
        x_distance_to_goal = (self.player_pos[0] - self.goal_pos[0]) / self.params.space_size[0]
        y_distance_to_goal = (self.player_pos[1] - self.goal_pos[1]) / self.params.space_size[1]

        x_distance_to_dead = (self.player_pos[0] - self.dead_zone_pos[0]) / self.params.space_size[0]
        y_distance_to_dead = (self.player_pos[1] - self.dead_zone_pos[1]) / self.params.space_size[1]

        x = self.player_pos[0] / self.params.space_size[0]
        y = self.player_pos[1] / self.params.space_size[1]

        actions = self.creature.input_observations_output_actions(np.array([x_distance_to_goal,y_distance_to_goal,x_distance_to_dead,y_distance_to_dead,x,y]).reshape(1,-1))
        #actions = self.creature.input_observations_output_actions(self.environment.flatten().reshape(1,-1))
        actions = actions.flatten()
        #print("actions",actions)
        self.action_and_reward(actions)



    def action_and_reward(self,actions):

        assert len(actions) == 2

        new_pos = copy.deepcopy(self.player_pos)
        if actions[0] > 0:
            new_pos[0] += 1
        else:
            new_pos[0] -= 1
        if actions[1] > 0:
            new_pos[1]  += 1
        else:
            new_pos[1] -= 1




        if self.get_object_in_space(new_pos) == DEAD_ZONE:
            self.creature.score_record(-1)
            self.game_over = 1

        if self.get_object_in_space(new_pos) == WALL:
            self.creature.score_record(-0.3)
            self.game_over = 1

        if self.get_object_in_space(new_pos) == GOAL:
            self.creature.score_record(1)
            self.game_over = 1

        if self.get_object_in_space(new_pos) == GROUND:
            self.set_player_pos(new_pos)
            self.step += 1
            if self.step > (self.params.space_size[0] + self.params.space_size[1]) :
                self.game_over = 1
                return
            if (self.player_pos[0] - self.goal_pos[0]) + (self.player_pos[1] - self.goal_pos[1]) < self.last_distance:
                self.creature.score_record(0.1)
                self.last_distance = (self.player_pos[0] - self.goal_pos[0]) + (self.player_pos[1] - self.goal_pos[1])
            else:
                self.creature.score_record(-0.1)



    def display(self):
        '''
        matplotlib movable figs.
        '''
        p.imshow(self.environment)
        p.show()

    def get_creature(self):
        return self.creature




def run_one_creature(exist_creature=None):
    if exist_creature == None:
        a_brain = Brain(layer_structure=[6,2])
        a = Creatures(a_brain)
        b = SingleCreatureDemo(SingleCreatureDemoParams(), creature=a)

    else:
        assert  isinstance(exist_creature, Creatures)
        b = SingleCreatureDemo(SingleCreatureDemoParams(), creature=exist_creature)
    while b.game_over != 1:

        b.run_output_observations_input_actions()

    print(b.creature.score)
    return b.creature

def run():
    strategy = HighScoreKeepStrategy()
    group = CreaturesGroup(strategy)
    for i in range(10):
        group.add(run_one_creature())
    for epoch in range(10):
        print(group.get_mean_std_of_score())
        creatures = group.evolve()
        group.clear()
        for c in creatures:
            group.add(run_one_creature(c))
    c = group.sort_by_score()
    good_creature = c[0]
    good_creature.save()
    print(good_creature.score)












if __name__ == '__main__':

    run()





# demo
# group = Group()
# def give_one_task_to_cpu(creaures):
# e = Environment()
# c = Creatures()
# while e.gameover != 1:
#   e.send_message(c)
#   c.send_message(e)
# c.get_reward(e.give_reward)
# group.add(c)

# if group.number > 10: c = group.evolve
# continue cpu...


