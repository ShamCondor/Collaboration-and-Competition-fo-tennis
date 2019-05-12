
#  Project 3: Collaboration and Competition


## MADDPG

Multi-agent DDPG (MADDPG) (Lowe et al., 2017) extends DDPG to an environment where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents.

In summary, MADDPG added three additional ingredients on top of DDPG to make it adapt to the multi-agent environment:

   -Centralized critic + decentralized actors;
   
   -Actors are able to use estimated policies of other agents for learning;
   
   -Policy ensembling is good for reducing variance.
   

### Hyper parameters for MADDPG

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 256        
GAMMA = 0.99           
TAU = 5e-3              
LR_ACTOR = 1e-3         
LR_CRITIC = 1e-3        
WEIGHT_DECAY = 0


### Neural Networks for MADDPG

Actor and Critic network models were defined in maddpg_model.py.

The Actor networks utilised two fully connected layers with 256 and 256 units with relu activation and tanh activation for the action space. 
The Critic networks utilised two fully connected layers with 256 and 256 units with relu activation. 



## Performance of the agent

![Alt text](https://github.com/Quertier/p3_collab-compet/blob/master/p3_maddpg.PNG)


## Future Improvements

As future works, we could replace DDPG with Distributed Distributional Deterministic Policy Gradients (D4PG) [https://arxiv.org/pdf/1804.08617.pdf].



