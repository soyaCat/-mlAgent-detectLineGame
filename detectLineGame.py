'''
약어 사전
mbc: marked boundary camera 의 약자
wfnliiocn: write_file_name_list_index_instead_of_correct_name 의 약자
'''


import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import datetime
import time
import math
from collections import deque
import os

import random
import CustomFuncionFor_mlAgent as CF
from PIL import Image
from tqdm import tqdm

game = "DetectLineGame.exe"
env_path = "./build/" + game
save_picture_path = "./made_data/"
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_model_path = "./saved_model/"+date_time+"_DQN/"
load_model_path = "./saved_model/"+"20210328-224721_DQN/model/model"
load_model = True
save_model = False

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=20.0, target_frame_rate=120, capture_frame_rate=120)
env = UnityEnvironment(file_name=env_path, side_channels=[channel])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log=None, ConversionDataType=ConversionDataType)

connection_test_count = 0
pre_stack_step_before_train = 2
train_count = 1000
test_count =0

max_episode_step_in_episode = 300
target_update_step = 10000
print_train_statues_interval_episode_count = 1
save_model_interval_episode_count = 50


write_file_name_list_index_instead_of_correct_name = False
list_index_for_main = 0
list_index_for_mbc0 = 1
list_index_for_mbc1 = 2
list_index_for_mbc2 = 3
list_index_for_mbc3 = 4
list_index_for_bc0 = 5
list_index_for_bc1 = 6
list_index_for_bc2 = 7
list_index_for_bc3 = 8
generate_main = True
generate_mbc0 = True
generate_mbc1 = True
generate_mbc2 = True
generate_mbc3 = True
generate_boundary_cam = True

state_size = [128,128,3]
action_size = 6

epsilon_init = 0.9
epsilon_min = 0.1
learning_rate = 0.00025
batch_size = 64
mem_maxlen = 50000
discount_factor = 0.9

class DQN_Network():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape = [None, state_size[0], state_size[1], state_size[2]], dtype = tf.float32)
        self.input_normalize = (self.input - (255.0/2) / (255.0/2))
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu,
                                          kernel_size=[8,8], strides=[4,4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu,
                                          kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu,
                                          kernel_size=[3,3], strides=[1,1], padding="SAME")
            self.flat = tf.layers.flatten(self.conv3)
            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)
        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DQN():
    def __init__(self):
        self.epsilon = epsilon_init
        self.action_size = action_size
        self.model = DQN_Network("Q")
        self.target_model = DQN_Network("target")
        self.memory = deque(maxlen = mem_maxlen)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.update_target()
        print("hello?")
        if load_model == True:
            print("Is?")
            self.Saver.restore(self.sess, load_model_path)
            print("work?")
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_size)
        else:
            state = state.reshape((1,state_size[0],state_size[1],state_size[2]))
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input:state})
            return np.asscalar(predict)

    def append_sample(self, dic1set):
        self.memory.append((dic1set.get('state'),
                            dic1set.get('action'),
                            dic1set.get('reward'),
                            dic1set.get('next_state'),
                            dic1set.get('done'),
                            ))

    def save_model(self):
        self.Saver.save(self.sess, save_model_path+"/model/model")

    def train_model(self, done):
        # 앱실론 값 감소
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 1/train_count

        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input:states})
        target_val = self.sess.run(self.target_model.Q_Out, feed_dict={self.target_model.input: next_states})
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor*np.amax(target_val[i])
        _, loss = self.sess.run([self.model.UpdateModel, self.model.loss], feed_dict={self.model.input:states, self.model.target_Q: target})

        return loss

    def update_target(self):
        for i in range(len(self.model.train_var)):
            self.sess.run(self.target_model.train_var[i].assign(self.model.train_var[i]))

    def Make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)
        Summary = tf.summary.FileWriter(logdir=save_model_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    def Write_Summary(self, reward, loss, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss:loss, self.summary_reward:reward}), episode)




class Get_Reward():
    def __init__(self):
        self.reward = 0
        self.target_bc_area = 0
        self.pre_target_hide_area = 0
        self.target_bc_index = -1

    def init_reward(self, mbc_dic, bc_dic):
        bc_sum_list = [np.sum(bc_dic[0]), np.sum(bc_dic[1]), np.sum(bc_dic[2]), np.sum(bc_dic[3])]
        self.target_bc_index = bc_sum_list.index(max(bc_sum_list))
        self.target_bc_area = bc_sum_list[self.target_bc_index]
        self.pre_target_hide_area = np.sum(bc_dic[self.target_bc_index])-np.sum(mbc_dic[self.target_bc_index])
        self.reward = 0
        print("인덱스:", self.target_bc_index)

    def update_reward(self, mbc_dic):
        # target_hide_area가 target_bc이면 라인을 맞춘 것, 0 넓이라면 라인을 전혀 못 맞춘것
        # target_hide_area가 커지는 방향이면 올바르게 찾는것, 작아지는 방향이면 올바르지 못하게 찾는 것
        target_hide_area = self.target_bc_area-np.sum(mbc_dic[self.target_bc_index])
        game_fin = False
        # print("\n")
        # print("시작")
        # print("index :", self.target_bc_index)
        # print("tba   :", self.target_bc_area)
        # print("tmba  :", np.sum(mbc_dic[self.target_bc_index]))
        # print("tha   :", target_hide_area)
        # print("ptha  :", self.pre_target_hide_area)

        if target_hide_area == 0:
            self.reward = -0.05
        elif target_hide_area == self.target_bc_area:
            self.reward = 1
            game_fin = True

        else:
            area_diff = int(target_hide_area) - int(self.pre_target_hide_area)
            # print("arad   :", area_diff)
            self.reward = area_diff/self.target_bc_area
            self.reward *= 10
            self.pre_target_hide_area = target_hide_area

        return self.reward, game_fin




def save_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    im = Image.fromarray(vis_observation_list[list_index].astype('uint8'), 'RGB')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')

def save_gray_numpy_file(append_name, list_index, wfnliiocn, episodeCount):
    target = ConversionDataType.Reduction_Dimention_for_grayIMG(vis_observation_list[list_index])
    im = Image.fromarray(target.astype('uint8'), 'L')
    if wfnliiocn == False:
        im.save(save_picture_path + str(episodeCount) + append_name + '.jpg')
    else:
        im.save(save_picture_path + str(list_index) + '.jpg')



if __name__ == '__main__':
    totalEpisodeCount = train_count + test_count
    Get_Reward = Get_Reward()
    DQN = DQN()


    for episodeCount in tqdm(range(connection_test_count)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        wfnliiocn = write_file_name_list_index_instead_of_correct_name
        if generate_main is True:
            save_numpy_file('_main', list_index_for_main, wfnliiocn, episodeCount)
        if generate_mbc0 is True:
            save_gray_numpy_file('_mbc0', list_index_for_mbc0, wfnliiocn, episodeCount)
        if generate_mbc1 is True:
            save_gray_numpy_file('_mbc1', list_index_for_mbc1, wfnliiocn, episodeCount)
        if generate_mbc2 is True:
            save_gray_numpy_file('_mbc2', list_index_for_mbc2, wfnliiocn, episodeCount)
        if generate_mbc3 is True:
            save_gray_numpy_file('_mbc3', list_index_for_mbc3, wfnliiocn, episodeCount)
        if generate_boundary_cam is True:
            save_gray_numpy_file('_bc0', list_index_for_bc0, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc1', list_index_for_bc1, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc2', list_index_for_bc2, wfnliiocn, episodeCount)
            save_gray_numpy_file('_bc3', list_index_for_bc3, wfnliiocn, episodeCount)
        action = [1, 0, 0, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()


    for episodeCount in tqdm(range(pre_stack_step_before_train)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        bc_dic = {
            0: vis_observation_list[list_index_for_bc0],
            1: vis_observation_list[list_index_for_bc1],
            2: vis_observation_list[list_index_for_bc2],
            3: vis_observation_list[list_index_for_bc3]}
        mbc_dic = {
            0: vis_observation_list[list_index_for_mbc0],
            1: vis_observation_list[list_index_for_mbc1],
            2: vis_observation_list[list_index_for_mbc2],
            3: vis_observation_list[list_index_for_mbc3]}
        Get_Reward.init_reward(mbc_dic, bc_dic)

        dic1set = {
            'state':0,
            'action':0,
            'reward':0,
            'next_state':0,
            'done':0}
        state = vis_observation_list[list_index_for_main]
        dic1set.update(state=state)
        episode_rewards = 0
        done = False
        rewards = []
        losses = []
        episode_step = 0

        while 1:
            episode_step += 1
            behavior_name = behavior_names[0]

            # 다음 행동을 계산한 후 유니티 환경에 적용한다.
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
            state = vis_observation_list[list_index_for_main]
            calculated_action_index = DQN.get_action(state)
            action = [2, 0, 0, 0, 0, 0, 0]
            action[calculated_action_index+1] = 1
            dic1set.update(action=action[1:])
            actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
            env.set_actions(behavior_name, actionTuple)
            env.step()

            # 다음 상태, 보상, 게임 종료 정보를 취득한다.
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
            next_state = vis_observation_list[list_index_for_main]
            dic1set.update(next_state=next_state)
            mbc_dic = {
                0: vis_observation_list[list_index_for_mbc0],
                1: vis_observation_list[list_index_for_mbc1],
                2: vis_observation_list[list_index_for_mbc2],
                3: vis_observation_list[list_index_for_mbc3]}
            reward, done = Get_Reward.update_reward(mbc_dic)
            dic1set.update(reward=reward, done=done)
            DQN.append_sample(dic1set)

            # state를 업데이트 한다.
            dic1set.update(state=next_state)
            episode_rewards += reward

            #타겟 네트워크 업데이트
            if episode_step % (target_update_step) == 0:
                DQN.update_target()

            # done == True인 경우 또는 step이 maxstep을 초과할 경우 while문을 탈출
            if done == True or episode_step % max_episode_step_in_episode == 0:
                break

        # while 문 탈출
        print("ep_step{} / episode: {} / ep_rewards: {:.2f} ".format(episode_step, episodeCount, np.mean(episode_rewards)))
        # 게임 초기화
        action = [1, 0, 0, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()


    total_step = 0
    for episodeCount in tqdm(range(train_count)):
        behavior_name = behavior_names[0]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
        bc_dic = {
            0: vis_observation_list[list_index_for_bc0],
            1: vis_observation_list[list_index_for_bc1],
            2: vis_observation_list[list_index_for_bc2],
            3: vis_observation_list[list_index_for_bc3]}
        mbc_dic = {
            0: vis_observation_list[list_index_for_mbc0],
            1: vis_observation_list[list_index_for_mbc1],
            2: vis_observation_list[list_index_for_mbc2],
            3: vis_observation_list[list_index_for_mbc3]}
        Get_Reward.init_reward(mbc_dic, bc_dic)

        dic1set = {
            'state':0,
            'action':0,
            'reward':0,
            'next_state':0,
            'done':0}
        state = vis_observation_list[list_index_for_main]
        dic1set.update(state=state)
        episode_rewards = 0
        done = False
        rewards = []
        losses = []
        episode_step = 0

        while 1:
            total_step += 1
            episode_step += 1
            behavior_name = behavior_names[0]

            # 다음 행동을 계산한 후 유니티 환경에 적용한다.
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
            state = vis_observation_list[list_index_for_main]
            calculated_action_index = DQN.get_action(state)
            action = [2, 0, 0, 0, 0, 0, 0]
            action[calculated_action_index+1] = 1
            dic1set.update(action=action[1:])
            actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
            env.set_actions(behavior_name, actionTuple)
            env.step()

            # 다음 상태, 보상, 게임 종료 정보를 취득한다.
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
            next_state = vis_observation_list[list_index_for_main]
            dic1set.update(next_state=next_state)
            mbc_dic = {
                0: vis_observation_list[list_index_for_mbc0],
                1: vis_observation_list[list_index_for_mbc1],
                2: vis_observation_list[list_index_for_mbc2],
                3: vis_observation_list[list_index_for_mbc3]}
            reward, done = Get_Reward.update_reward(mbc_dic)
            dic1set.update(reward=reward, done=done)
            DQN.append_sample(dic1set)

            # state를 업데이트 한다.
            dic1set.update(state=next_state)
            episode_rewards += reward

            # loss값을 구하다가 학습을 진행한다
            if episode_step % max_episode_step_in_episode == 0:
                done = True
            loss = DQN.train_model(done)
            losses.append(loss)

            #타겟 네트워크 업데이트
            if total_step % (target_update_step) == 0:
                DQN.update_target()

            # done == True인 경우 에피소드를 종료한다.
            if done == True:
                break

        # while 문 탈출
        rewards.append(episode_rewards)
        # 게임 진행 상황 출력 및 텐서보드에 보상과 손실함수값 기록
        if episodeCount % print_train_statues_interval_episode_count == 0 and episodeCount != 0:
            print("step{} / episode: {} / reward{:.2f} / loss: {:.4f}/ epsilon{:.3f}".format(total_step, episodeCount, np.mean(rewards), np.mean(losses), DQN.epsilon))
            DQN.Write_Summary(np.mean(rewards), np.mean(losses), episodeCount)
            rewards = []
            losses = []

        # 네트워크 모델 저장
        if episodeCount % save_model_interval_episode_count == 0 and episodeCount != 0:
            DQN.save_model()
            print("Save Model {}".format(episodeCount))

        # 게임 초기화
        action = [1, 0, 0, 0, 0, 0, 0]
        actionTuple = ConversionDataType.ConvertList2DiscreteAction(action, behavior_name)
        env.set_actions(behavior_name, actionTuple)
        env.step()

    env.close()
