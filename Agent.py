import matplotlib.pyplot as plt
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import FileSystemTools as fst
from collections import namedtuple
from matplotlib.colors import LinearSegmentedColormap
import traceback as tb
import json

Experience = namedtuple('exp_tup',('s','a','r','s_next'))


class DQN(nn.Module):

	def __init__(self,D_in,H,D_out,NL_fn=torch.tanh,softmax=False):
		super(DQN, self).__init__()

		self.lin1 = nn.Linear(D_in,H)
		self.lin2 = nn.Linear(H,D_out)
		self.NL_fn = NL_fn
		self.softmax = softmax

	def forward(self, x):
		x = self.lin1(x)
		#x = F.relu(x)
		#x = torch.tanh(x)
		x = self.NL_fn(x)
		x = self.lin2(x)
		if self.softmax:
			x = torch.softmax(x,dim=1)
		return(x)


class Agent:

	def __init__(self, **kwargs):

		self.agent_class = kwargs.get('agent_class',None)
		self.agent_class_name = self.agent_class.__name__
		self.agent = self.agent_class(**kwargs)

		self.params = self.agent.getPassedParams()

		#Eventually, it would be good to make it so it only actually adds to the filename
		#the parameters that were specifically passed to it, and adds the default ones
		#to the log file.
		self.params['Agent'] = self.agent_class_name

		self.params['epsilon'] = kwargs.get('epsilon',0.9)
		self.params['eps_decay'] = kwargs.get('eps_decay', 0.999)
		self.params['eps_min'] = kwargs.get('eps_min', 0.02)

		self.params['N_steps'] = kwargs.get('N_steps',10**3)
		self.params['N_batch'] = int(kwargs.get('N_batch',60))
		self.params['N_hidden_layer_nodes'] = int(kwargs.get('N_hidden_layer_nodes',100))
		self.params['target_update'] = int(kwargs.get('target_update',500))
		self.params['double_DQN'] = kwargs.get('double_DQN',False)

		self.params['features'] = kwargs.get('features','DQN')
		self.params['NL_fn'] = kwargs.get('NL_fn','tanh')
		self.params['loss_method'] = kwargs.get('loss_method','L2')

		self.params['advantage'] = kwargs.get('advantage', True)
		self.params['beta'] = kwargs.get('beta',0.1)

		self.dir = kwargs.get('dir','misc_runs')
		self.date_time = kwargs.get('date_time',fst.getDateString())
		#self.base_fname = fst.paramDictToFnameStr(self.params) + '_' + self.date_time
		self.base_fname = fst.paramDictToFnameStr(self.params) + '_' + self.date_time
		self.img_fname = fst.combineDirAndFile(self.dir, self.base_fname + '.png')
		self.log_fname = fst.combineDirAndFile(self.dir, 'log_' + self.base_fname + '.txt')
		self.reward_fname = fst.combineDirAndFile(self.dir, 'reward_' + self.base_fname + '.txt')
		self.weights_hist_fname = fst.combineDirAndFile(self.dir, 'weights_hist_' + self.base_fname + '.txt')
		self.model_fname = fst.combineDirAndFile(self.dir, 'model_' + self.base_fname + '.model')
		self.optimizer_fname = fst.combineDirAndFile(self.dir, 'optimizer_' + self.base_fname + '.model')

		##### These are params we don't need in the basename.

		self.params['alpha'] = kwargs.get('alpha',10**-1)
		self.params['ACER'] = kwargs.get('ACER', False)

		self.params['exp_buf_len'] = int(kwargs.get('exp_buf_len',10000))
		self.params['gamma'] = kwargs.get('gamma',1.0)
		self.params['clamp_grad'] = kwargs.get('clamp_grad',False)

		self.figsize = kwargs.get('figsize', (12, 8))
		self.R_tot_hist = []


		self.initLearningParams()

		#self.createFigure()

		self.params['save_params_to_file'] = kwargs.get('save_params_to_file', True)
		if self.params['save_params_to_file']:
			fst.writeDictToFile(self.params, self.log_fname)


	def initLearningParams(self):

		self.R_tot = 0
		self.scheduler = None

		self.exp_pos = 0
		self.N_actions = self.agent.N_actions #Should change this to a getter fn at some point.

		#self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = torch.device('cpu')
		#print('\nusing device:',self.device)

		self.dtype = torch.float32
		torch.set_default_dtype(self.dtype)

		if str(self.device) == 'cpu':
			torch.set_default_tensor_type(torch.FloatTensor)
		if str(self.device) == 'cuda':
			torch.set_default_tensor_type(torch.cuda.FloatTensor)


		NL_fn_dict = {'relu':F.relu, 'tanh':torch.tanh, 'sigmoid':F.sigmoid}

		self.weights_history = []

		if self.params['features'] == 'linear':
			self.N_feats = self.getStateVec().shape[0]
			print('state vec shape: ', self.N_feats)
			self.w_Q = torch.zeros((self.N_feats, self.N_actions), device=self.device, dtype=self.dtype, requires_grad=True)
			#self.theta_pi = torch.zeros(self.N_feats, device=self.device, dtype=self.dtype, requires_grad=True)
			#self.optimizer = optim.RMSprop([self.w_Q])
			#self.optimizer = optim.Adam([self.w_Q])
			self.optimizer = optim.SGD([self.w_Q], lr=10**-2)
			self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99999, last_epoch=-1)



		if self.params['features'] == 'DQN':
			#This is using a "target" Q network and a "policy" Q network, with
			#experience replay.

			D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions

			NL_fn = NL_fn_dict[self.params['NL_fn']]

			self.policy_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)
			self.target_NN = DQN(D_in,H,D_out,NL_fn=NL_fn)

			if str(self.device) == 'cuda':
				self.policy_NN.cuda()
				self.target_NN.cuda()

			self.target_NN.load_state_dict(self.policy_NN.state_dict())
			self.target_NN.eval()
			#self.optimizer = optim.RMSprop(self.policy_NN.parameters())
			self.optimizer = optim.Adam(self.policy_NN.parameters())
			self.experiences = []

			self.episode = self.DQNepisode


		if self.params['features'] == 'AC':

			D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions

			NL_fn = NL_fn_dict[self.params['NL_fn']]

			#The "actor" is the policy network, the "critic" is the Q network.
			self.actor_NN = DQN(D_in, H, D_out, NL_fn=NL_fn, softmax=True)

			if self.params['advantage']:
				self.critic_NN = DQN(D_in, H, 1, NL_fn=NL_fn)
			else:
				self.critic_NN = DQN(D_in, H, D_out, NL_fn=NL_fn)

			if str(self.device) == 'cuda':
				self.actor_NN.cuda()
				self.critic_NN.cuda()

			self.actor_optimizer = optim.RMSprop(self.actor_NN.parameters())
			self.critic_optimizer = optim.RMSprop(self.critic_NN.parameters())
			self.experiences = []

			if self.params['ACER']:
				self.episode = self.ACERepisode
			else:
				self.episode = self.ACepisode





	########### Functions for different episode types.


	def DQNepisode(self, show_plot=False, save_plot=False, save_NN_info=False):

		if show_plot:
			self.showFig()

		self.initEpisode()

		s = self.getStateVec()

		try:
			for i in range(self.params['N_steps']):

				self.updateEpsilon()

				self.updateFrozenQ(i)

				a = self.epsGreedyAction(self.policy_NN, s)
				r, s_next = self.iterate(a)

				self.updateR(r, i)

				a_next = self.epsGreedyAction(self.policy_NN, s_next)

				self.addExperience(Experience(s,a,r,s_next))

				if len(self.experiences)>=2*self.params['N_batch']:

					#Get random batch
					batch_Q_samples = sample(self.experiences,self.params['N_batch'])
					experiences = Experience(*zip(*batch_Q_samples))

					states = torch.stack(experiences.s)
					actions = experiences.a
					rewards = torch.stack(experiences.r)
					states_next = torch.stack(experiences.s_next)

					Q_cur = self.forwardPass(self.policy_NN, states)[list(range(len(actions))),actions]

					if self.params['double_DQN']:
						actions_next = self.forwardPass(self.policy_NN, states_next).argmax(dim=1)
						Q_next = self.forwardPass(self.target_NN, states_next)[list(range(len(actions))), actions_next]
					else:
						Q_next = torch.max(self.forwardPass(self.target_NN, states_next), dim=1)[0]


					if self.params['loss_method'] == 'smoothL1':
						TD0_error = F.smooth_l1_loss(Q_cur,(rewards + self.params['gamma']*Q_next).detach())
					if self.params['loss_method'] == 'L2':
						TD0_error = (rewards + self.params['gamma']*Q_next - Q_cur).pow(2).sum()

					self.optimizer.zero_grad()
					TD0_error.backward()

					self.optimizer.step()

				s = s_next

				if show_plot:
					self.plotAll()
					self.fig.canvas.draw()

			exit_code = 0

		except:
			print('\nrun stopped, saving model and plots. Reason for stop:')
			print(tb.format_exc())
			exit_code = 1
			#del self.agent

		'''if save_plot:
			self.plotAll()
			plt.savefig(self.img_fname)

		plt.close('all')'''

		if save_NN_info:
			self.saveModel()
			self.saveOptimizer()
			self.saveRewardCurve()

		return_dict = {
		'R_avg' : self.R_tot/self.params['N_steps'],
		'exit_code' : exit_code
		}

		print('self.R_tot/N_steps: {:.2f}'.format(return_dict['R_avg']))

		return(return_dict)


	def ACepisode(self, show_plot=True, save_plot=False):

		if show_plot:
			self.showFig()

		self.initEpisode()

		s = self.getStateVec()
		a = self.softmaxAction(self.actor_NN, s)

		for i in range(self.params['N_steps']):

			r, s_next = self.iterate(a)
			a_next = self.softmaxAction(self.actor_NN, s_next)

			self.updateR(r, i)

			if self.params['advantage']:
				critic_cur = self.forwardPass(self.critic_NN, s)
				critic_next = self.forwardPass(self.critic_NN, s_next)
				critic_value = (r + self.params['gamma']*critic_next.detach() - critic_cur.detach())
			else:
				critic_cur = self.forwardPass(self.critic_NN, s)[a]
				critic_next = self.forwardPass(self.critic_NN, s_next)[a_next]
				critic_value = critic_cur.detach()

			if self.params['loss_method'] == 'smoothL1':
				#There seems to be a bug in this, where it says that the "derivative for target (the 2nd arg) isn't implemented".
				TD0_error = F.smooth_l1_loss(critic_cur, r + self.params['gamma']*critic_next)
			if self.params['loss_method'] == 'L2':
				TD0_error = (r + self.params['gamma']*critic_next - critic_cur).pow(2).sum()


			pi = self.forwardPass(self.actor_NN, s)
			iota = 10**-6
			pi = (pi + iota)/sum(pi + iota)
			#This sum is HIGHER for more entropy. So we want to put it in J with a negative, to maximize it. Err...seems backwards.
			entropy = -self.params['beta']*sum(pi*torch.log(pi))
			J = -(critic_value*torch.log(pi[a]) + entropy)


			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()

			J.backward(retain_graph=True)
			TD0_error.backward()

			self.actor_optimizer.step()
			self.critic_optimizer.step()

			s = s_next
			a = a_next

			if show_plot:
				self.plotAll()
				self.fig.canvas.draw()


		if save_plot:
			self.plotAll()
			plt.savefig(self.img_fname)

		plt.close('all')

		#self.saveModel()
		self.saveRewardCurve()

		print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
		return(self.R_tot/self.params['N_steps'])


	def ACERepisode(self,show_plot=True,save_plot=False):

		self.R_tot = 0

		if show_plot:
			self.showFig()

		self.initEpisode()

		s = self.getStateVec()

		for i in range(self.params['N_steps']):

			a = self.softmaxAction(s)
			r, s_next = self.iterate(a)

			self.R_tot += r.item()
			self.R_tot_hist.append(self.R_tot/(i+1))

			if i%int(self.params['N_steps']/10) == 0:
				print('iteration {}, R_tot/i = {:.3f}'.format(i,self.R_tot_hist[-1]))



			self.addExperience(Experience(s,a,r,s_next))

			if len(self.experiences)>=2*self.params['N_batch']:


				#Get random batch
				batch_samples = sample(self.experiences,self.params['N_batch'])
				experiences = Experience(*zip(*batch_samples))

				states = torch.stack(experiences.s)
				actions = experiences.a
				rewards = torch.stack(experiences.r)
				states_next = torch.stack(experiences.s_next)

				V_cur = self.value_NN(states)
				V_next = self.value_NN(states_next)


				a_next = self.softmaxAction(s_next)

				iota = 10**-4
				pi = self.actor_NN(states)
				pi = torch.softmax(pi + iota, dim=1)

				if np.isnan(pi.detach().numpy()).any():
					print('nan entry, stopping')
					print('state: ',s)
					print('pi: ',pi)
					exit(0)

				if self.params['loss_method'] == 'smoothL1':
					#There seems to be a bug in this, where it says that the "derivative for target (the 2nd arg) isn't implemented".
					TD0_error = F.smooth_l1_loss(V_cur, r + self.params['gamma']*V_next)
				if self.params['loss_method'] == 'L2':
					TD0_error = (r + self.params['gamma']*V_next - V_cur).pow(2).sum()


				#This sum is HIGHER for more entropy. So we want to put it in J with a negative, to maximize it. Err...seems backwards.
				entropy = -self.params['beta']*torch.sum(pi*torch.log(pi),dim=1)/(pi.shape[0])
				J = (-(r + self.params['gamma']*V_next.detach() - V_cur.detach())*torch.log(pi[list(range(len(actions))),actions]) - entropy).sum()

				self.actor_optimizer.zero_grad()
				self.critic_optimizer.zero_grad()

				J.backward(retain_graph=True)
				TD0_error.backward()

				if self.params['clamp_grad']:
					for param in self.actor_NN.parameters():
						param.grad.data.clamp_(-1, 1)
					for param in self.critic_NN.parameters():
						param.grad.data.clamp_(-1, 1)

				self.actor_optimizer.step()
				self.critic_optimizer.step()
				self.value_optimizer.step()

			s = s_next

			if show_plot:
				self.plotAll()
				self.fig.canvas.draw()


		if save_plot:
			self.plotAll()
			plt.savefig(self.img_fname)

		plt.close('all')

		self.saveModel()
		self.saveRewardCurve()

		print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
		return(self.R_tot/self.params['N_steps'])


	def linearQLearnEpisode(self, show_plot=True, save_plot=False):

		if show_plot:
			self.showFig()

		self.initEpisode()

		s = self.getStateVec()
		a = self.linEpsGreedyAction(s)

		#10**-2 works well with eps=0, crappy with eps=0.9...
		alpha = 10**-2
		for i in range(self.params['N_steps']):

			self.updateWeightsList(self.w_Q.detach().reshape(-1).tolist())

			self.updateEpsilon()
			self.scheduler.step()

			r, s_next = self.iterate(a)

			self.updateR(r, i)

			a_next = self.linEpsGreedyAction(s_next)

			Q_cur = torch.matmul(self.w_Q.t(), s)[a]
			Q_next = torch.matmul(self.w_Q.t(), s_next).detach()[a_next]

			TD0_error = (r + self.params['gamma']*Q_next - Q_cur).pow(2).sum()

			'''TD0_error.backward()
			with torch.no_grad():
				self.w_Q -= alpha*self.w_Q.grad
				self.w_Q.grad.zero_()'''

			self.optimizer.zero_grad()
			TD0_error.backward()
			self.optimizer.step()

			s = s_next
			a = a_next


			if show_plot:
				self.plotAll()
				self.fig.canvas.draw()

		if save_plot:
			self.plotAll()
			plt.savefig(self.img_fname)

		plt.close('all')

		self.saveRewardCurve()
		if self.N_feats == 6:
			header_str = '\t'.join(['x_U', 'x_D', 'x_L', 'x_R', 'y_U', 'y_D', 'y_L', 'y_R', 'vx_U', 'vx_D', 'vx_L', 'vx_R', 'vy_U', 'vy_D', 'vy_L', 'vy_R', 'tx_U', 'tx_D', 'tx_L', 'tx_R', 'ty_U', 'ty_D', 'ty_L', 'ty_R'])
		elif self.N_feats == 4:
			header_str = '\t'.join(['x_U', 'x_D', 'x_L', 'x_R', 'y_U', 'y_D', 'y_L', 'y_R', 'tx_U', 'tx_D', 'tx_L', 'tx_R', 'ty_U', 'ty_D', 'ty_L', 'ty_R'])
		header_str += '\n{}x{}'.format(self.w_Q.shape[0], self.w_Q.shape[1])
		self.saveWeightsHistory(header=header_str)

		print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
		return(self.R_tot/self.params['N_steps'])




	######### Functions for updating values periodically.


	def updateFrozenQ(self, iteration):

		if iteration%self.params['target_update']==0 and iteration>self.params['N_batch']:
			if self.params['features'] == 'DQN':
				self.target_NN.load_state_dict(self.policy_NN.state_dict())
			if self.params['features'] == 'AC':
				self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())


	def updateEpsilon(self):
		# This will have the effect of always trying to move eps to eps_min, even if
		# you set eps=0.
		self.params['epsilon'] = (self.params['epsilon'] - self.params['eps_min'])*self.params['eps_decay'] + self.params['eps_min']


	def addExperience(self, experience):
		if len(self.experiences) < self.params['exp_buf_len']:
			self.experiences.append(None)
		self.experiences[self.exp_pos] = experience
		self.exp_pos = (self.exp_pos + 1) % self.params['exp_buf_len']


	def updateR(self, r, iteration):
		self.R_tot += r.item()
		#self.R_tot_hist.append(self.R_tot/(iteration+1))
		self.R_tot_hist.append(self.R_tot)

		if iteration%max(1, int(self.params['N_steps']/10)) == 0:
			print('iteration {}, R_tot/i = {:.3f}'.format(iteration, self.R_tot_hist[-1]))
			if self.scheduler is not None:
				print('LR: {:.4f}'.format(self.optimizer.param_groups[0]['lr']))


	def updateWeightsList(self, weights):

		self.weights_history.append(weights)



	########### Functions for interacting with networks.


	def softmaxAction(self, NN, state_vec):
		pi_vals = self.forwardPass(self.actor_NN, state_vec).detach()
		m = Categorical(pi_vals)
		return(m.sample().item())


	def forwardPass(self, NN, state_vec):
		#This will automatically figure out if it's a single tensor or a batch,
		#and if it's a single, unsqueeze it to the right dimension and then squeeze it
		#again after. It assumes that the network only has a single input dimension, though
		#(so it won't handle it if the input shape for the network is 3x5 or something).
		assert len(state_vec.shape)>=1, 'len of state vec is <1, error.'

		if len(state_vec.shape)==1:
			output = NN(state_vec.unsqueeze(dim=0)).squeeze()
		else:
			output = NN(state_vec)

		return(output)


	def greedyAction(self, NN, state_vec):
		return(self.forwardPass(NN, state_vec).argmax().item())


	def epsGreedyAction(self, NN, state_vec):
		if random()>self.params['epsilon']:
			return(self.greedyAction(NN, state_vec))
		else:
			return(self.getRandomAction())


	def getRandomAction(self):
		return(randint(0, self.N_actions-1))


	def linEpsGreedyAction(self, s):

		if random()>self.params['epsilon']:
			return(self.linGreedyAction(s))
		else:
			return(self.getRandomAction())


	def linGreedyAction(self, state_vec):
		return(torch.matmul(self.w_Q.t(), state_vec).argmax())


	def valueFunction(self, state_vec):

		if self.params['features']=='linear':
			return(torch.matmul(self.w_Q.t(), state_vec))

		if self.params['features']=='DQN':
			return(self.forwardPass(self.policy_NN, state_vec))



	########### These are functions for interacting with the agent object class.


	def iterate(self, a):
		r, s_next = self.agent.iterate(a)
		return(torch.tensor(r, device=self.device, dtype=self.dtype), torch.tensor(s_next, device=self.device, dtype=self.dtype))


	def initEpisode(self):
		self.agent.initEpisode()


	def resetStateValues(self):
		self.agent.resetStateValues()


	def getStateVec(self):
		return(torch.tensor(self.agent.getStateVec(), device=self.device, dtype=self.dtype))


	def getReward(self):
		return(torch.tensor(self.agent.reward(),device=self.device,dtype=self.dtype))



	############ Functions for saving and loading.


	def loadModel(self, **kwargs):
		# So this will match them, but it shouldn't be too bad.
		fname = kwargs.get('fname', None)
		print('\nloading model from {}'.format(fname))
		self.policy_NN.load_state_dict(torch.load(fname))
		self.target_NN.load_state_dict(self.policy_NN.state_dict())


	def loadOptimizer(self, **kwargs):
		fname = kwargs.get('fname', None)
		print('\nloading optimizer from {}'.format(fname))
		self.optimizer.load_state_dict(torch.load(fname))


	def loadParams(self, **kwargs):
		fname = kwargs.get('fname', None)
		print('\nloading params from {}'.format(fname))
		with open(fname, 'r') as f:
			NN_params = json.load(f)

		self.params['epsilon'] = NN_params['epsilon']
		self.exp_pos = NN_params['exp_pos']


	def loadExperiences(self, **kwargs):
		fname = kwargs.get('fname', None)
		print('\nloading experiences from {}'.format(fname))
		with open(fname, 'r') as f:
			exp_dict = json.load(f)

		temp_exp = []
		for i, e_dict in exp_dict.items():
			temp_exp.append(Experience(torch.tensor(e_dict['s']), e_dict['a'], torch.tensor(e_dict['r']), torch.tensor(e_dict['s_next']) ))

		self.experiences = temp_exp




	def saveModel(self, **kwargs):
		fname = kwargs.get('fname', self.model_fname)
		print('\nsaving model to {}'.format(fname))
		torch.save(self.policy_NN.state_dict(), fname)


	def saveOptimizer(self, **kwargs):
		fname = kwargs.get('fname', self.optimizer_fname)
		print('\nsaving optimizer to {}'.format(fname))
		torch.save(self.optimizer.state_dict(), fname)


	def saveParams(self, **kwargs):
		fname = kwargs.get('fname', None)
		print('\nsaving NN params to {}'.format(fname))
		NN_params = {
		'epsilon' : self.params['epsilon'],
		'exp_pos' : self.exp_pos
		}
		with open(fname, 'w+') as f:
			json.dump(NN_params, f, indent=4)



	def saveExperiences(self, **kwargs):
		fname = kwargs.get('fname', None)
		print('\nsaving experiences to {}'.format(fname))
		temp_exp = {}
		for i, e in enumerate(self.experiences):
			temp_exp[i] = {
			's' : e.s.tolist(),
			'a' : e.a,
			'r' : e.r.item(),
			's_next' : e.s_next.tolist()
			}

		with open(fname, 'w+') as f:
			json.dump(temp_exp, f, indent=4)



	def loadModelPlay(self, model_fname, show_plot=True, save_plot=False, make_gif=False, N_steps=10**3, load_params=False, zero_eps=True):

		# Right now, I think this will only work for DQN...maybe it makes more sense to make it part of AgentTools?
		# Then again, I'd like the ability to train it with one agent class (e.g., PWA_radial) and then
		# use the trained network on another (e.g., Robot).


		if load_params:
			# This is very much assuming that the filename scheme stays the same, that the
			# model file and log file use the same date timestamp, etc.
			dir = fst.dirFromFullPath(model_fname)
			local_model_fname = fst.fnameFromFullPath(model_fname)
			local_log_fname = local_model_fname.replace('model_', 'log_').replace('.model', '.txt')
			log_fname = fst.combineDirAndFile(dir, local_log_fname)
			print('Loading params from log file: {}'.format(log_fname))

			log_dict = fst.readFileToDict(log_fname)
			for k,v in log_dict.items():
				if k in self.params.keys():
					print('setting self.params[\'{}\'] from {} to {}'.format(k, self.params[k], v))
					self.params[k] = v


		self.policy_NN.load_state_dict(torch.load(model_fname))


		if zero_eps:
			self.params['epsilon'] = 0

		self.R_tot = 0

		if show_plot:
			self.showFig()

		self.initEpisode()


		for i in range(N_steps):

			if i%int(N_steps/10) == 0:
				print('iteration ',i)

			self.updateEpsilon()

			s = self.getStateVec()
			a = self.epsGreedyAction(self.policy_NN, s)

			r, s_next = self.iterate(a)

			self.updateR(r, i)

			if show_plot:
				self.plotAll()
				self.fig.canvas.draw()

			if make_gif:
				self.plotAll()
				plt.savefig(fst.combineDirAndFile(self.dir, str(i) + '.png'))


		if save_plot and not make_gif:
			self.plotAll()
			plt.savefig(self.img_fname)

		plt.close('all')

		return(self.R_tot/N_steps)


	def saveRewardCurve(self):
		print('\nsaving reward history to {}'.format(self.reward_fname))
		np.savetxt(self.reward_fname, self.R_tot_hist[10:], fmt='%.4f')


	def saveWeightsHistory(self, header=None):
		if header is None:
			np.savetxt(self.weights_hist_fname, self.weights_history, fmt='%.4f')
		else:
			np.savetxt(self.weights_hist_fname, self.weights_history, header=header, fmt='%.4f')



	############ Functions for plotting and drawing.

	def plotAll(self):
		self.drawState()
		self.plotStateParams()
		self.plotRtot()

		#self.plotWeights()
		if self.params['features'] == 'DQN':
			pass


	def drawState(self):
		self.agent.drawState(self.ax_state)


	def plotWeights(self):

		sv = self.getStateVec().detach().numpy().tolist()
		target_pos = sv[4:]

		if (target_pos != self.last_target_pos) or (self.last_target_pos is None):

			self.ax_wQ.clear()
			self.last_target_pos = target_pos
			N_disc = 50
			lims = self.agent.xlims
			pos = np.array([[[x,y] for y in np.linspace(lims[0],lims[1],N_disc)] for x in np.linspace(lims[0],lims[1],N_disc)])
			v = np.expand_dims(np.full((N_disc,N_disc),0),axis=2)
			sv = self.getStateVec()
			xt = np.expand_dims(np.full((N_disc,N_disc),sv[4]),axis=2)
			yt = np.expand_dims(np.full((N_disc,N_disc),sv[5]),axis=2)

			states = np.concatenate((pos,v,v,xt,yt),axis=2)

			states = torch.tensor(states,dtype=self.dtype)
			output = self.forwardPass(self.policy_NN, states)

			best_actions = (torch.argmax(output,dim=2)).detach().numpy()
			#best_actions = (torch.max(output,dim=2)[0]).detach().numpy()

			if self.col_bar is not None:
				self.col_bar.remove()

			col_plot = self.ax_wQ.matshow(best_actions.T,cmap=self.cm,origin='lower')
			#col_plot = self.ax_wQ.matshow(best_actions,cmap='Reds',origin='lower')
			self.ax_wQ.set_xlabel('x')
			self.ax_wQ.set_ylabel('y')
			self.col_bar = self.fig.colorbar(col_plot,ax=self.ax_wQ, ticks=[0,1,2,3], boundaries=np.arange(-.5,4.5,1))
			self.col_bar.ax.set_yticklabels(['U','D','L','R'])
			xt = sv[4]
			yt = sv[5]

			target = plt.Circle(((xt-lims[0])*N_disc,(yt-lims[0])*N_disc), 2.5*N_disc/20.0, color='black')
			self.ax_wQ.add_artist(target)





			self.ax_wQ2.clear()
			self.last_target_pos = target_pos
			N_disc = 40
			lims = self.agent.xlims
			pos = np.array([[[x,y] for y in np.linspace(lims[0],lims[1],N_disc)] for x in np.linspace(lims[0],lims[1],N_disc)])
			v = np.expand_dims(np.full((N_disc,N_disc),0),axis=2)
			sv = self.getStateVec()
			xt = np.expand_dims(np.full((N_disc,N_disc),sv[4]),axis=2)
			yt = np.expand_dims(np.full((N_disc,N_disc),sv[5]),axis=2)

			states = np.concatenate((pos,v,v,xt,yt),axis=2)

			states = torch.tensor(states,dtype=self.dtype)
			output = self.forwardPass(self.policy_NN, states)

			#best_actions = (torch.argmax(output,dim=2)).detach().numpy()
			max_Q = (torch.max(output,dim=2)[0]).detach().numpy()

			if self.col_bar2 is not None:
				self.col_bar2.remove()

			col_plot2 = self.ax_wQ2.matshow(max_Q.T,cmap='Reds',origin='lower')
			#col_plot2 = self.ax_wQ2.matshow(max_Q,cmap='Reds',origin='lower')
			self.ax_wQ2.set_xlabel('x')
			self.ax_wQ2.set_ylabel('y')
			self.col_bar2 = self.fig.colorbar(col_plot2,ax=self.ax_wQ2)
			xt = sv[4]
			yt = sv[5]

			target = plt.Circle(((xt-lims[0])*N_disc,(yt-lims[0])*N_disc), 2.5*N_disc/20.0, color='black')
			self.ax_wQ2.add_artist(target)


	def plotRtot(self):
		self.ax_R_tot.clear()
		self.ax_R_tot.plot(self.R_tot_hist[8:])


	def plotStateParams(self):
		self.agent.plotStateParams([self.ax_state_params1,self.ax_state_params2,self.ax_state_params3,self.ax_state_params4])


	def createFigure(self):

		self.fig, self.axes = plt.subplots(3, 3, figsize=self.figsize)
		self.ax_state = self.axes[0,0]
		self.ax_state_params1 = self.axes[0,1]
		self.ax_state_params2 = self.axes[0,2]
		self.ax_state_params3 = self.axes[1,2]
		self.ax_state_params4 = self.axes[1,1]

		self.ax_wQ = self.axes[2,1]
		self.ax_wQ2 = self.axes[2,0]
		self.ax_theta_pi = self.axes[1,1]

		#self.ax_loc_vals = self.axes[2,0]
		self.ax_R_tot = self.axes[2,2]
		self.col_bar = None
		self.col_bar2 = None

		self.cm = LinearSegmentedColormap.from_list('my_cm', ['tomato','dodgerblue','seagreen','orange'], N=4)

		self.last_target_pos = None
		self.plotAll()


	def showFig(self):
		plt.show(block=False)



	#################################### scrap

	def scrap(self):

		pass

		#for a while I was getting an error in AC because I think some values of pi were going to 0, so log(pi)
		#was an error, so I did this ghetto thing with iota and checked here...

		#iota = 10**-4
		#pi = torch.softmax(pi + iota, dim=0)
		'''if np.isnan(pi.detach().numpy()).any():
			print('nan entry, stopping')
			print('state: ',s)
			print('pi: ',pi)
			exit(0)'''


		'''
		#This is probably better to do? I'm not sure if it needs it to backward() the things...
		probs2 = torch.squeeze(self.actor_NN(torch.unsqueeze(s_next, dim=0)))
		m2 = Categorical(probs2)
		a_next = m2.sample()'''

		#old version of softmaxaction:
		'''p = pi_vals.detach().numpy() #+ iota
		#p = torch.softmax(torch.tensor(p), dim=0).detach().numpy().tolist()
		#print(p)
		#p=torch.softmax(iota + pi_vals.detach(),dim=0).numpy()
		action = np.random.choice(list(range(len(pi_vals))),p=p)
		return(action)'''

#
