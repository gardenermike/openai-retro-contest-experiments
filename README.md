# Experiments for OpenAI Retro Contest

![Example video](./example.gif "An example round of sonic")

## Summary
The [OpenAI Retro Contest](https://contest.openai.com/) was an opportunity to use the new [Retro](https://github.com/openai/retro) environment to test transfer learning for reinforcement learning algorithms by playing classic Sonic the Hedgehog games. The environment is played fundamentally like human players: the agent sees only the game pixels, and interacts by taking actions available on the game controller.

I spent most of my time outside of work and sleep during May of 2018 tinkering with algorithms for the contest. This repo holds code that I used for my experiments. There are several starting points here for exploring some of the ideas further.

## Algorithms
The contest came with a set of [baseline](https://github.com/openai/retro-baselines) implementations. My work was all influenced by these baselines.

The three baseline implementations are:

### Rainbow
Rainbow is a Deep Q-Learning (DQN) algorithm I have seen described as a "kitchen sink" approach: it combines a variety of techniques used to improve DQN results together into a high-performing whole. [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithms attempt to maximize future _reward_ by learning from past experience. "Q" represents the function that returns the expected reward of an action. Honestly, it's an awkward name, but... whatever. "Deep" Q-Learning approximates the Q function using a neural network. For more information about Rainbow, the original paper is on [Arxiv](https://arxiv.org/abs/1710.02298).

### PPO2
[PPO2](https://blog.openai.com/openai-baselines-ppo/) is a high-performing policy gradient algorithm. Instead of optimizing the _reward_ per-action, as DQN algorithms do, policy gradient algorithms optimize an explicit _policy_ mapping the observed game state to an action. The policy is iteratively improved during training to take actions that lead to a higher reward. PPO (and the extension, PPO2) uses a clever clipping approach to prevent sudden updates to the policy while also avoiding expensive calculation of divergence between old and new policies, allowing for both stable and efficient training.

### JERK (Just Enough Retained Knowledge) 
[JERK](https://arxiv.org/abs/1804.03720) is an algorithm introduced as part of the retro contest. The idea is simple: try random sequences of runs and jumps, and replay the sequence with highest reward as the contest time limit approaches. In a deterministic environment, JERK could eventually achieve perfect performance. The retro contest environment is _not_ deterministic, however. The contest environment randomizes the number of steps an action is applied, so a long sequence of statically-defined actions will, over time, become more and more likely to lose sync with the game.

### And the outlier, *Evolution Strategies*
Evolution Strategies, unlike the baseline algorithms, explores by making random changes to the _policy_ rather than taking random _actions_. It is typically slow and inefficient, but can be massively parallelized across multiple processors and machines, unlike algorithms with more shared state. I explored using Evolution Strategies to find good weights for just the flat layer of my PPO2 policy after pretraining the convolutional layers. I think that the approach has a lot of promise, but I do not currently have access to sufficient compute resources to find a good policy in a reasonable amount of time (less than weeks or months).

## Results Summary
tl;dr: Pretrained Rainbow works best, but a JERK/PPO2 hybrid with LSH memory came close.

|Algorithm|Best score (contest test set)|
|---------|-----------------------------|
|Rainbow|**4406.35**|
|PPO2|2410.79|
|JERK|2678.63|
|JERK/PPO2 hybrid|3631.06|
###### * All algorithms but JERK were pretrained on training levels

## Detailed discussion

The [retro contest blog post](https://blog.openai.com/retro-contest/) describes the impetus for the contest: a notable improvement in PPO2 performance with weights pre-trained on different levels of the game.

![Contest performance graph](https://blog.openai.com/content/images/2018/03/plot_all_human_3-28b.png "Contest performance graph")

### Early success
Immediately I noticed that Rainbow had the best performance out-of-the-box, and that pretraining might help. I [tweaked](https://github.com/gardenermike/openai-retro-contest-experiments/blob/master/rainbow/rainbow_agent_train.py) the baseline implementation to play training levels at random serially: every 5000 game steps, the current game level would be terminated and replaced with another. After training for around 5,000,000 steps, I built a version that would load the weights and I [gave it a go](https://github.com/gardenermike/openai-retro-contest-experiments/blob/master/rainbow/docker-build/rainbow_agent.py).
In that run was both the great success and tragedy of my contest experience: the docker image built with those pretrained weights was my best-performing of the contest.
I did add not only the weights from the pretrained model, but a replay buffer of 100,000 frames from training levels. This experience buffer allowed the agent to start making informed actions immediately, and likely explains my better Rainbow performance over the published baseline.

### A series of less-successful experiments in Rainbow
Emboldened by my early success, I moved forward. I attempted:
* _Longer training._
  Longer training appears to have overfit the training levels, and performance was never better than the first try.
* _Stacking more than four frames._
  Performance here was abysmal. I believe that I had a bug in my implementation that I noticed some time later, so bumping up the frame count might still be worth a try.
* _An attention mechanism in the model._
  This was slow as mud. The approach was inspired by [Temporal Convolutional Networks](https://arxiv.org/abs/1608.08242). Maybe some day when computation speed catches up with the algorithm, this approach could become practical. I tried submitting an implementation, and it was cut off at less than 50% complete when it reached the hard contest time limit.
* _Using the convolutional layer weights pretrained with PPO2._
  I applied PPO2 to all 47 training levels at once for a combined 50 million timesteps. The resulting weights from the convolutional layer had learned to "see" the relevant characteristics in sonic quite well, so I tried bringing them over to my Rainbow implementation. The resulting performance was on par with my original results, with a highest score of 3461.98. My lowest score on the original Rainbow pretraining was 3411.67, by comparison.

### PPO2
PPO2 had (and still has) promise. It performed well in the contest baselines, and was around twice as fast as Rainbow, making it a pleasure to work with. The tooling around the baseline algorithms was also nice: I was able to set up an arbitrary number of parallel agents working together. I initially started with the serial approach I used with Rainbow, with poor results. The training never really improved. I believe this was because a _policy_ is hard to stabilize without a consistent environment, while _value_ is predictable across environments. I moved on to parallel training, using cloud instances to train on all 47 Sonic training levels at once. Over time, the "explained variance" in the model rose to approximately 96%, suggesting that the convolutional layers had learned what to expect from the game. However, the policy plateaued. I think that my next approach would be to improve exploration in the PPO2 model by using a noise layer similar to Rainbow. I have read that the noise layer leads to better exploration than the entropy bonus in the baseline implementation.

Other PPO2 experiments:
* Using a mobilenet model instead of the simpler three-layer convolutional network also used in the Rainbow baseline implementation.
  The explained variance took longer to climb, but the overall performance never showed any benefit. I believe that the core weakness in my PPO2 experiments is exploration, so a stronger visualizing ability was not helpful.
* Combining with JERK.
  Instead of using random exploration in JERK, I alternated between random exploration and using my pre-trained PPO2 policy. This approach showed a lot of promise, surpassing the weaker Rainbow runs.

### JERK
Who knew working with a jerk could be so rewarding?
The simplicity of JERK means that it runs fast. On simple levels like GreenHillZone.Act1, JERK can achieve human-level performance during some runs with only a  minute or less of training. It is amazing to watch. However, on levels with time sensitivity, such as SpringYardZone.Act1, the deterministic behavior of JERK simply can't deal with the stochasticity of the contest environment.

I experimented a lot with JERK. I made some immediate modifications:
* After replaying a successful run, my agent will continue to play if the level is incomplete
* Instead of using purely random play, my agent immediately begins after the first random run to replay with mutation. In each level, the agent begins with the most successful prior sequence and mutates it by:
  * Excising internal steps from the run
  * Truncating some actions from the end of the sequence
  * Randomly mutating some actions, weighted to more mutation at the end of the sequence
* Replay without mutation is swapped in heavily near the contest completion.

With the above performance improvements, JERK performs a little worse on simpler levels but has much improved performance on more difficult levels.

The idea of memory in JERK was a potent one, so I continued to explore the limits of the algorithm.
I implemented two major improvments:
* I brought a pre-trained PPO2 model into my JERK agent. This model is used for a fraction of the play that would originally be random.
* I used the output of the PPO2 policy prior to categorization as an embedding, together with [locality-sensitive hashing (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing), to look up similar prior frames and replay the highest-performing actions from those frames. This is not a new idea; it was previously explored in the [Neural Episodic Control paper](https://arxiv.org/abs/1703.01988).

With these additional improvements, I was able to bring JERK performance to a similar level to Rainbow (3631.06), but not quite good enough to best it.

I think that my JERK hybrid still has room to grow. Some ideas for future exploration:
1. The LSH memory contains many frames that are almost exact or exact duplicates, which can cause the memory to only return a single action per-frame. Pre-filtering the history while periodically building the memory could solve this problem.
1. The PPO2 agent is not currently learning in parallel with the JERK training. Allowing simultaneous training could allow for more improvement.
1. Rainbow could be used instead of PPO2 as the backing algorithm. An additional benefit to Rainbow is that high-performing JERK sequences could be added to Rainbow's replay buffer to improve performance of Rainbow. DeepMind's recent [Mix & Match paper](https://arxiv.org/abs/1806.01780) illustrated the power of expert sequences in improving Rainbow performance.

### Evolution Strategies
I didn't hear about the contest until May, although it had started over a month earlier. With a few days left and no run that had put me in a position above 12 in the leaderboard, I tried a final idea in the background while I ran some last Rainbow experiments.

OpenAI published a [blog post](https://blog.openai.com/evolution-strategies/), [paper](https://arxiv.org/abs/1703.03864), and [code](https://github.com/openai/evolution-strategies-starter) in March of 2017 detailing use of Evolution Strategies (ES) to solve problems in a relatively challenging domains. Like many algorithms underlying recent advances in machine learning, ES has been around for a while, but only recently has the compute power been available to make it practical for real problems.

Evolution Strategies randomizes the _policy_ of an agent rather than exploring with random _actions_. Since the policy represented by even a modest network may contain millions of weights, searching for useful combinations in that space could easily be futile. However, I found that that the convolutional layers of my PPO2 policy seemed to have been able to learn the important characteristics of the game, and the weakness was exploration. With that in mind, I tweaked the OpenAI starter implementation to modify only the fully-connected layer of my PPO2 policy, and trained beginning with four workers, then attempted to expand outward.
The OpenAI implementation had gone somewhat stale in the past year: the Amazon machine images were out of date, the python libraries were out of date, and the code was targeted only to the MuJoCo environments. In addition, being only one guy and not an organization with more resources, I was limited in the number of EC2 instances I could spin up (20). In OpenAI's paper, they described going over 7000 CPU cores! I got an implementation mostly working on a single machine, but the distributed implementation had some surmountable problems that I just didn't have time to work through in the couple of days I had remaining. I also tried scaling up to a larger single instance, but ran into performance problems that I also have not yet diagnosed.

I ran a little over 100 training passes across all 47 training levels across 4 workers.

Despite the limited training, I did see some interesting results. In particular, in some of the runs saw significant spikes in score, and I saw an average score of ~200 per level, which while not impressive, is certainly non-zero.

I have pushed my code [here](https://github.com/gardenermike/evolution-strategies-starter). The [policies file](https://github.com/gardenermike/evolution-strategies-starter/blob/master/es_distributed/policies.py) contains a Sonic policy that tweaks the pretrained PPO2 policy's fully-connected layer. Expect bugs!

The learned weights from my runs are [in this repo](https://github.com/gardenermike/openai-retro-contest-experiments/tree/master/ppo2_evolution_strategies/es_master_17351).

## Conclusion
I don't expect to reach the top three in the contest, and am probably outside the top 10. However, there is ample room for further exploration here, potentially leading to not just high scores in Sonic, but potentially generalizable algorithms for real-world problems.
I think that the most promising directions to take forward include:
* Improving exploration in PPO2. The simplicity of the PPO2 algorithm makes it very appealing.
* Combining JERK with Rainbow to improve the performance of both.
* Expanding and tidying the ES implementation.
* Improving the LSH memory mechanism used in the PPO2/JERK hybrid.

Good luck!
