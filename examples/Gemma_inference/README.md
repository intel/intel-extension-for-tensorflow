# Gemma Inference Best Known Method for Intel-Extension-For-Tensorflow on Intel GPU

## Introduction
Gemma is a family of open-weights Large Language Model (LLM) by Google DeepMind, based on Gemini research and technology. For more detail information, please refer to [Gemma/keras](https://www.kaggle.com/models/google/gemma).

This example shows how to run Keras3 implementation of Gemma inference with Intel® Extension for TensorFlow* on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series
 - Intel® Data Center GPU Flex Series 170
 
## Prerequisites
### Request Access
Follow [Gemma-setup](https://www.kaggle.com/code/nilaychauhan/get-started-with-gemma-using-kerasnlp#Gemma-setup) to apply access permission of kaggle.

### Clone the Repository(Only for Accuracy Check) <a name="clone-repo"></a>
```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git lm_eval 
cd lm_eval
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
git apply ../gemma.patch
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md#prepare)

### Setup Running Environment
* Setup for GPU
```bash
./pip_set_env.sh
```
Note: This Gemma keras3 implementation requires TensorFlow >= 2.16.1 and Intel® Extension for TensorFlow* >= 2.16.0.0.

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)

### Executes the Example with Python API
#### Model Default Parameters
| **Parameter** | **Default Value** |
| :---: | :--- |
| **model** | gemma_2b |
| **dtype** | float32 |
| **input-tokens** | 32 |
| **max-new-tokens** | 32 |
| **num-beams** | 1 |
| **num-iter** | 10 |
| **num-warmup** | 3 |
| **batch-size** | 1 |

#### FP32 Inference
```
python run_generation.py \
  --model gemma_2b       \
  --dtype float32        \
  --input-tokens 32      \
  --max-new-tokens 32
```

#### BF16 Inference
```
python run_generation.py \
  --model gemma_2b       \
  --dtype bfloat16       \
  --input-tokens 32      \
  --max-new-tokens 32
```

#### Accuracy Check
Note: If you want check accuracy, please [clone the repository](#clone-repo) first.
```
python main.py \
  --model gemma \
  --model_args model_name=gemma_2b_en,dtype=float32,num_beams=1 \
  --tasks openbookqa \
  --no_cache
```

## Example Output
With successful execution, it will print out the following results:

```
Iteration: 0, Time: xxx sec
Iteration: 1, Time: xxx sec
Iteration: 2, Time: xxx sec
Iteration: 3, Time: xxx sec
Iteration: 4, Time: xxx sec
Iteration: 5, Time: xxx sec
Iteration: 6, Time: xxx sec
Iteration: 7, Time: xxx sec
Iteration: 8, Time: xxx sec
Iteration: 9, Time: xxx sec

 ---------- Summary: ----------
Inference latency: xxx sec.
Output: ["It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept evolved into a more complex game, with more mechanics. I'll go through them in the next section. The game is a lot more complex than it looks. It has a lot of moving parts, and it's not easy to explain. I'll try to explain it as best as I can. The game is played in a 2D top-down view. The player controls a single pasta, which is spawned on the player's plate. The player can move the pasta around, and can also order other pastas to be spawned. The player can also order the pasta to 'eat' other pastas, which"].
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
