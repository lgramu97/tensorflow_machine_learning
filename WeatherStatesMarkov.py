'''

We can find these probabilities from large datasets or may already have these values. 
We'll run through an example in a second that should clear some things up,
but let's discuss the  components of a markov model.

**States:** In each markov model we have a finite set of states. 
These states could be something like "warm" and "cold" or "high" and "low" or even "red",
 "green" and "blue". These states are "hidden" within the model, 
 which means we do not direcly observe them.

**Observations:** Each state has a particular outcome or observation associated with it
 based on a probability distribution. An example of this is the following: 
    *On a hot day Tim has a 80% chance of being happy and a 20% chance of being sad.*

**Transitions:** Each state will have a probability defining the likelyhood of transitioning
 to a different state. An example is the following: 
 *a cold day has a 30% chance of being followed by a hot day and a 
 70% chance of being follwed by another cold day.*

To create a hidden markov model we need.
- States
- Observation Distribution
- Transition Distribution

For our purpose we will assume we already have this information available as 
we attempt to predict the weather on a given day.

'''


import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

'''
We will model a simple weather system and try to predict the temperature on each day given
the following information.
1. Cold days are encoded by a 0 and hot days are encoded by a 1.
2. The first day in our sequence has an 80% chance of being cold.
3. A cold day has a 30% chance of being followed by a hot day.
4. A hot day has a 20% chance of being followed by a cold day.
5. On each day the temperature is
 normally distributed with mean and standard deviation 0 and 5 on
 a cold day and mean and standard deviation 15 and 10 on a hot day.

 Same problem : https://es.wikipedia.org/wiki/Modelo_oculto_de_M%C3%A1rkov#Ejemplo_de_utilizaci%C3%B3n
'''

def run():
    tfd = tfp.distributions  # making a shortcut for later on
    initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  # Refer to point 2 above
    transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                    [0.2, 0.8]])  # refer to points 3 and 4 above
    observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above
    # the loc argument represents the mean and the scale is the standard devitation

    model = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=7) #Steps is how many time we want to predict.

    mean = model.mean()

    # due to the way TensorFlow works on a lower level we need to evaluate part of the graph
    # from within a session to see the value of this tensor

    # in the new version of tensorflow we need to use tf.compat.v1.Session() 
    # rather than just tf.Session()
    with tf.compat.v1.Session() as sess:  
        print('Average temperature per day:' ,mean.numpy())

    #So that's it for the core learning algorithms in TensorFlow. 
    #Hopefully you've learned about a few interesting tools that are easy to use! 
    #To practice I'd encourage you to try out some of these algorithms on different datasets.

if __name__ == '__main__':
    run()