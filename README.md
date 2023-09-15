# A simpy Simulation code to simulate ticket issue and passenger boardings at a train station.

## Problem Statement

Consider the Malleswaram suburban train station with two platforms. The first platform
serves the trains departing towards Bangalore City and the second platform serves the
trains departing towards Tumkur. Passengers board the train on a first-come-first-serve
basis. 

The trains have a limited total capacity of 120 seats. The available capacity in
any train when it arrives at the Malleswaram station is modelled as the random variable
𝐶 ≈ 𝑈(15, 35). 

The inter-arrival times of trains travelling towards Bangalore City and
Tumkur are modelled by the random variables 𝑇𝐵 ≈ exp(14) and 𝑇𝑇 ≈ exp(25) respectively.

The actual observations of some 5000 inter-arrival times of passengers at the Malleswaram
station for travel towards Bangalore City and Tumkur are given separately.

Tickets are issued at the Malleswaram station by a single ticket clerk and two automated
ticketing machines. The time taken to issue tickets by the ticketing clerk and the auotomated 
ticketing machines are modelled by the random variables 𝑇𝑚 ≈ exp (1) and
𝑇𝑇 ≈ exp (0.7) respectively. 

The time to failure of the automated ticketing machine is
the random variable 𝑇𝑓 ≈ Norm𝜇 = 1000, 𝜎 = 200) and the repair time is the random
variable 𝑇𝑟 ≈ exp(𝜇 = 30).

### Output needed 
Develop a discrete event simulation model in SimPy to simulate this system to answer the
following questions.

1. Should the number of automated ticketing machines be increased so that the average
waiting times for purchasing tickets is no more than 5 minutes for at least 90 percent
of the travellers?
2. What is the average waiting time of travellers from the time of their arrival at the
station till they board a train? Should the frequency of trains be increased?
